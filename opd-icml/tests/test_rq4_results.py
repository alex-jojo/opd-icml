import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from zipfile import ZipFile

import pytest
import torch

from test_parameter_tokens import ROOT, make_actor, make_data, objective, opd, optimizer_update, study_module
from test_parameter_sampling import cache_logits
from test_rq2_motion import configure

sys.path.insert(0, str(ROOT/'scripts'))
import rq4_results as rq4
import run_h200_experiment as h200
import run_h200_suite as suite
import cpu_resources
from package_results import package_results
from test_h200_experiments import checkpoint_fixture
sys.path.insert(0, str(ROOT/'runtime/verl/trainer/ppo'))
from opd_replication_checkpoint import SharedEarlyStop


def cached_loss(actor, data):
    with torch.no_grad():
        return objective(actor, data)[data.batch['response_mask'].bool()].mean().item()


def test_second_update_gain_on_same_cached_batch(tmp_path, monkeypatch):
    torch.manual_seed(16)
    actor, data = make_actor(), make_data()
    configure(actor, tmp_path, rounds=53, reuse=2)
    actor.config.ppo_epochs = 1
    actor.config.ppo_mini_batch_size = len(data)
    actor.config.policy_loss.update(jump_reuse_updates=2)
    actor.config.policy_loss['replication17']['enabled'] = True
    actor.config.policy_loss['gradient_study']['parameter_tokens'] = dict(
        enabled=True, mode='kl_stratified', sample_size=4)
    monkeypatch.setattr(type(data), 'to', lambda self, device: self, raising=False)
    teacher = data.batch['opd_teacher_top16_logits'].clone()
    directory = tmp_path/'training_records/rq4'
    history = []
    def commit():
        history.append(cached_loss(actor, data))
        norm = torch.nn.utils.clip_grad_norm_(actor.actor_module.parameters(), 1.)
        actor.actor_optimizer.step()
        return norm
    actor._optimizer_step = commit
    for step in range(1, 54):
        data.meta_info['global_steps'] = step
        cache_logits(actor, data)
        metrics = opd.update(actor, data)
        assert metrics['actor/opd_optimizer_steps'] == [2.]
        final = cached_loss(actor, data)
        path = directory/f'reuse_round_{step:03d}.json'
        assert path.exists() == (step % 5 == 0 or step == 53)
        if path.exists():
            row = rq4.read(path)
            assert row['optimizer_updates'] == [2*step-1, 2*step]
            assert row['valid_tokens'] == 5
            assert row['kl_before_first'] == pytest.approx(history[-2], abs=1e-6)
            assert row['kl_after_first'] == pytest.approx(history[-1], abs=1e-6)
            assert row['kl_after_second'] == pytest.approx(final, abs=1e-6)
            assert row['second_gain'] == pytest.approx(history[-1]-final, abs=1e-6)
    assert len(list(directory.glob('reuse_round_*.json'))) == 11
    assert len(history) == 106
    assert torch.equal(data.batch['opd_teacher_top16_logits'], teacher)


def test_adaptive_window_published_with_decision(tmp_path):
    torch.manual_seed(15)
    actor, data = make_actor(), make_data()
    configure(actor, tmp_path, adaptive=True)
    # Make the three-window rule fire deterministically at its first eligible step.
    actor.config.policy_loss['replication17']['adaptive_stability'].update(
        adjacent_overlap=.01, window_overlap=.01, p10_principal_cos2=.001, max_energy_gap=1.)
    obs = study_module.StudyReplication(actor)
    for step in range(1, 16):
        obs.begin(data, None, step)
        optimizer_update(actor, data)
        obs.after_inner(1)
        stable = obs.finish([0.], [0.])['replication/adaptive_stop']
        assert bool(stable) == (step == 15)
    records = [rq4.read(p) for p in sorted((tmp_path/'training_records/rq4').glob('early_stop_*.json'))]
    assert [r['step'] for r in records] == [5, 10, 15]
    assert [r['decision'] for r in records] == ['continue', 'continue', 'stable']
    assert records[-1]['steps'] == [5, 10, 15] and len(records[-1]['adjacent']) == 2
    assert 'energy_gap' in records[-1]['anchor'] and records[-1]['config']['interval'] == 5
    assert records[1]['checks'] is None and all(records[-1]['checks'].values())


def test_suite_rq4_handoff_scores_and_zip(tmp_path, monkeypatch):
    cpus = list(range(64))
    monkeypatch.setattr(cpu_resources, 'allowed_cpu_ids', lambda: cpus)
    monkeypatch.setattr(cpu_resources, 'cpu_quota', lambda: None)
    plan = suite.build_suite(suite.parser().parse_args(['--output', str(tmp_path/'suite')]),
                             {}, cpus, 512)
    paths = []
    base_runs = []
    early_evals = []
    import code_benchmarks
    monkeypatch.setattr(code_benchmarks, 'read_manifest', lambda *_: {})
    code_grades = []
    for queue in [*plan['small_queues'], plan['large']]:
        for job in queue:
            args = h200.parser().parse_args(job['command'][2:])
            strategy = args.rq4_strategy
            run = h200.project_absolute(job['output'])
            q = h200.build_plan(args, {})
            cap, split, final = (53, 2, 53) if strategy == 'reuse' else (106, 1, 106)
            q['resolved_training'] = dict(rounds_cap=cap, updates_per_round=split)
            window = run/'training/training_records/rq4'

            def launch(command, env, log):
                if command == q['train']:
                    assert not (run/'training').exists()
                    actor = checkpoint_fixture(run/'training', q['gpu_count'], final)
                    rq4.save_json(actor.parent/'opd_optimizer_updates.json', dict(rollout_round=final, optimizer_updates_done=final*split))
                    if strategy == 'full':
                        observer = SharedEarlyStop(actor.parent.parent, 0, 5)
                        for step in range(1, final + 1):
                            stable = args.scale in ('4b', 'deepseek-coder-33b') and step == 40
                            if step % 5 == 0:
                                rq4.save_json(window/f'early_stop_{step:03d}.json', dict(step=step, stable=stable))
                            observer.observe(step, step, stable, step == final, lambda path, _: shutil.copytree(actor, path))
                elif 'verl.model_merger' in command:
                    source = command[command.index('--local_dir')+1]
                    assert source.endswith((f'global_step_{final}/actor', 'early_stop/actor'))
                    merged = h200.project_absolute(command[command.index('--target_dir')+1])
                    merged.mkdir(parents=True)
                    (merged/'config.json').write_text('{}')
                    (merged/'weights.safetensors').touch()
                elif 'scripts/evaluate.py' in command:
                    values = dict(zip(rq4.BENCHMARKS, (.5, .2, .1, .3, .4)))
                    values['amc23'] = 1.
                    output = h200.project_absolute(command[command.index('--output_file')+1])
                    if output.parent.name == 'early-stop':
                        early_evals.append((args.scale, strategy))
                        values['math500'] = .25
                    rq4.save_json(output.with_suffix('.summary.json'), {
                        k: dict(problems=i+1, samples=8*(i+1), mean_at_8=v) for i,(k,v) in enumerate(values.items())})
                    output.write_text('{}\n')
                elif 'scripts/evaluate_code.py' in command:
                    output = h200.project_absolute(command[command.index('--output-dir')+1])
                    if '--generate-only' in command:
                        model = h200.project_absolute(command[command.index('--model-path')+1])
                        assert (model/'config.json').exists()
                        for name in rq4.CODE_BENCHMARKS:
                            rq4.save_json(output/name/'generations.jsonl', {'responses': ['a']*8})
                    else:
                        assert '--grade-only' in command and env['CUDA_VISIBLE_DEVICES'] == ''
                        code_grades.append((args.scale, strategy, output.parent.name))
                        value = .5 if output.parent.name == 'early-stop' else .75
                        for name in rq4.CODE_BENCHMARKS:
                            assert (output/name/'generations.jsonl').exists()
                        scores = {name: dict(problems=4, samples=32, **{'pass@8': value})
                                  for name in rq4.CODE_BENCHMARKS}
                        rq4.save_json(output/'summary.json', dict(metric='pass@8', benchmarks=scores))
                elif 'scripts/evaluate_base.py' in command:
                    assert '--no-package' in command and strategy == 'full'
                    assert command[command.index('--scale') + 1] == args.scale
                    target = run/'eval/base'
                    if '--grade-only' in command:
                        assert env['CUDA_VISIBLE_DEVICES'] == ''
                        for name in rq4.CODE_BENCHMARKS:
                            assert (target/'code_pass8'/name/'generations.jsonl').is_file()
                    else:
                        base_runs.append((args.scale, strategy))
                    if q['task'] == 'code':
                        if '--generate-only' in command:
                            for name in rq4.CODE_BENCHMARKS:
                                rq4.save_json(target/'code_pass8'/name/'generations.jsonl', {'responses':['a']*8})
                            return
                        scores = dict(pass_at_8=dict.fromkeys(rq4.CODE_BENCHMARKS, .25), two_benchmark_mean_pass_at_8=.25)
                    else:
                        scores = dict(strict_last_box=dict(mean_at_8=dict.fromkeys(rq4.BENCHMARKS, .1),
                                                           five_benchmark_mean_at_8=.1))
                    rq4.save_json(target/'summary.json', dict(scale=args.scale, strategy='base', model=q['models']['student'],
                                  rollout_rounds=0, optimizer_updates=0, training_wall_seconds=None, **scores))
            monkeypatch.setattr(h200, 'run_command', launch)
            h200.execute(args, q, {})
            if q['task'] == 'code':
                assert not (window/'summary.json').exists()
                grade_args = h200.parser().parse_args(job['grade'][2:])
                h200.execute(grade_args, h200.build_plan(grade_args, {}), {})
            saved = rq4.read(window/'summary.json')
            assert saved['optimizer_updates'] == final*split and saved['rollout_rounds'] == final
            assert saved['stop_reason'] == ('stable' if strategy == 'early-stop' else 'budget')
            if q['task'] == 'code':
                assert saved['two_benchmark_mean_pass_at_8'] == .75
            else:
                assert saved['five_benchmark_mean_at_8'] == pytest.approx(.3)
            assert saved['training_wall_seconds'] > 0
            paths.append(run)
    assert len(set(paths)) == len(paths) == 14
    assert len(code_grades) == 7
    assert early_evals == [('4b', 'full')]
    assert sorted(base_runs) == sorted((scale, 'full') for scale in h200.EXPERIMENTS)
    rq4.save_comparison(plan)
    root = h200.project_absolute(plan['output'])
    combined = rq4.read(root/'rq4_comparison.json')
    assert set(combined) == set(h200.EXPERIMENTS)
    code_early = combined['deepseek-coder-33b']['early-stop']
    assert code_early['two_benchmark_change_pp'] == -25 and code_early['rollout_rounds'] == 40
    assert set(code_early['responses_files']) == set(rq4.CODE_BENCHMARKS)
    early = combined['4b']['early-stop']
    assert early['source_strategy'] == 'full' and early['rollout_rounds'] == early['optimizer_updates'] == 40
    assert early['stop_reason'] == 'stable' and early['five_benchmark_change_pp'] == pytest.approx(-5)
    assert 0 < early['training_wall_seconds'] <= combined['4b']['full']['training_wall_seconds']
    budget = combined['1.7b']['early-stop']
    assert budget['stop_reason'] == 'budget' and budget['uses_full_endpoint']
    assert budget['five_benchmark_change_pp'] == budget['training_time_saved_fraction'] == 0
    assert budget['responses_file'] == 'eval/final_mean8.jsonl'
    for arms in combined.values():
        for strategy in rq4.STRATEGIES:
            row = arms[strategy]
            assert row['training_time_saved_fraction'] == pytest.approx(1-row['training_wall_seconds']/arms['full']['training_wall_seconds'])
    baseline = combined['4b']['base']
    assert baseline['five_benchmark_change_pp'] == pytest.approx(-20)
    assert baseline['scoring_mode'] == 'strict_last_box' and baseline['training_time_saved_fraction'] is None
    for scale, arms in combined.items():
        assert set(arms) == {'full', 'early-stop', 'reuse', 'base'}
        if h200.EXPERIMENTS[scale].task == 'code':
            assert arms['base']['two_benchmark_change_pp'] == -50
    with ZipFile(package_results(root)) as archive:
        assert 'suite/rq4_comparison.json' in archive.namelist()
        assert sum(n.endswith('/rq4/summary.json') for n in archive.namelist()) == 14
        assert sum(n.endswith('/rq4/early_stop_summary.json') for n in archive.namelist()) == 7
        assert 'suite/4b/eval/early-stop/final_mean8.jsonl' in archive.namelist()
        for path in code_early['responses_files'].values():
            assert 'suite/deepseek-coder-33b/' + path in archive.namelist()
        assert sum(n.endswith('/eval/base/summary.json') for n in archive.namelist()) == 7


def test_shared_checkpoint_is_first_committed_model_and_full_keeps_its_basis(tmp_path):
    states = {}
    saved_steps = []
    for mode in ('ordinary', 'shared'):
        torch.manual_seed(15)
        actor, data = make_actor(), make_data()
        run = tmp_path/mode
        configure(actor, run, rounds=20, adaptive=mode == 'shared')
        cfg = actor.config.policy_loss['replication17']['adaptive_stability']
        cfg.update(snapshot_only=mode == 'shared', adjacent_overlap=.01, window_overlap=.01,
                   p10_principal_cos2=.001, max_energy_gap=1.)
        observer = study_module.StudyReplication(actor)
        selection = SharedEarlyStop(run/'checkpoints', 0, 5)
        def save(path, step):
            saved_steps.append(step)
            Path(path).mkdir()
            torch.save(actor.actor_module.state_dict(), Path(path)/'model.pt')
        for step in range(1, 21):
            if mode == 'shared' and step == 16:
                observer = study_module.StudyReplication(actor)
                data.meta_info['opd_early_stop_selected'] = True
            observer.begin(data, None, step)
            optimizer_update(actor, data)
            observer.after_inner(1)
            metrics = observer.finish([0.], [0.])
            if step >= 10:
                assert observer.reference['frozen'] and observer.reference['cumulative_to'] == 10
            if mode == 'shared':
                selection.observe(step, step, metrics['replication/adaptive_stop'], step == 20, save)
            if step in (15, 20):
                states[mode, step] = {k:v.detach().clone() for k,v in actor.actor_module.state_dict().items()}
    assert saved_steps == [15]
    assert not (tmp_path/'shared/training_records/rq4/early_stop_020.json').exists()
    saved = torch.load(tmp_path/'shared/checkpoints/early_stop/actor/model.pt', weights_only=True)
    for key in saved:
        assert torch.equal(saved[key], states['ordinary', 15][key])
        assert torch.equal(states['shared', 20][key], states['ordinary', 20][key])
    assert any(not torch.equal(saved[key], states['shared', 20][key]) for key in saved)


def test_shared_snapshot_resume_and_failed_save(tmp_path):
    def fail(*args):
        raise RuntimeError('save failed')
    selection = SharedEarlyStop(tmp_path, 0, 5)
    with pytest.raises(RuntimeError, match='save failed'):
        selection.observe(15, 15, True, False, fail)
    assert not selection.marker.exists()
    selection.observe(15, 15, True, False, lambda *_: None)
    restored = SharedEarlyStop(tmp_path, 15, 5)
    restored.observe(20, 20, True, False, fail)
    assert rq4.read(restored.marker)['step'] == 15
    replay = SharedEarlyStop(tmp_path, 10, 5)
    assert not replay.marker.exists()
    replay.observe(20, 20, True, True, fail)
    assert rq4.read(replay.marker)['uses_full_endpoint']


def test_prefix_timing_handles_retries_and_missing_history():
    timing = dict(complete_history=True, attempts=[
        dict(started_at_unix=100, ended_at_unix=120, seconds=20),
        dict(started_at_unix=500, ended_at_unix=560, seconds=60)])
    assert rq4.prefix_wall_seconds(timing, 510) == 30
    assert rq4.prefix_wall_seconds(timing, 110) == 10
    assert rq4.prefix_wall_seconds(timing, 300) is None
    assert rq4.prefix_wall_seconds(dict(timing, complete_history=False), 110) is None


def test_eval_preserves_old_full_runs_and_reads_new_shared_run(tmp_path):
    args = h200.parser().parse_args(['--scale', '4b', '--output', str(tmp_path)])
    assert h200.build_plan(args, {})['shared_early_stop']
    args.stage = 'eval'
    for shared in (False, True):
        timing = dict(scale='4b', strategy='full')
        if shared:
            timing['shared_early_stop'] = True
        rq4.save_json(tmp_path/'rq4_training_time.json', timing)
        plan = h200.build_plan(args, {})
        assert plan['shared_early_stop'] == shared
        assert ('--rq4-strategy' in plan['train']) == shared


def test_early_stop_rejects_unconfirmed_partial_checkpoint(tmp_path):
    actor = checkpoint_fixture(tmp_path, 1, 20)
    rq4.save_json(actor.parent/'opd_optimizer_updates.json', dict(rollout_round=20, optimizer_updates_done=20))
    with pytest.raises(ValueError, match='confirmed stability stop'):
        rq4.endpoint(tmp_path/'checkpoints', tmp_path, 'early-stop', 106, 1)


@pytest.mark.parametrize('scale', h200.EXPERIMENTS)
@pytest.mark.parametrize('strategy', rq4.STRATEGIES)
def test_matched_real_hydra_config(scale, strategy):
    from hydra import compose, initialize_config_dir
    args = h200.parser().parse_args(['--scale', scale, '--rq4-strategy', strategy])
    cmd = h200.build_plan(args, {})['train']
    dry = json.loads(subprocess.check_output(cmd+['--dry-run'], cwd=ROOT, encoding='utf-8'))
    overrides = [s for s in shlex.split(dry['command']) if s.startswith('++')]
    search = 'hydra.searchpath='+json.dumps(['file://'+(ROOT/'runtime/verl/trainer/config').as_posix()], ensure_ascii=False)
    with initialize_config_dir(config_dir=str(ROOT/'configs'/scale), version_base=None):
        name = 'code-paired-full' if h200.EXPERIMENTS[scale].task == 'code' else 'paired-full'
        cfg = compose(config_name=name, overrides=[search, *overrides])
    policy = cfg.actor_rollout_ref.actor.policy_loss
    assert policy.jump_reuse_updates == policy.replication17.reuse == (2 if strategy == 'reuse' else 1)
    assert policy.replication17.rounds == cfg.trainer.opd_pause_after_rollout_rounds == cfg.trainer.total_training_steps
    assert cfg.trainer.total_training_steps*policy.jump_reuse_updates == 106
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert policy.gradient_study.parameter_tokens.sample_size == 64 and policy.fixed_heldout_probe_size == 0
    assert policy.replication17.enabled and policy.replication17.mode == 'full'
    assert policy.replication17.freeze_step == 10 and policy.replication17.rank == 128
    assert policy.gradient_study.enabled and policy.gradient_study.validation_interval == 10
    assert policy.gradient_study.subset_alignment_every_update
    assert list(policy.gradient_study.geometry_steps) == [10, 30, 60, 100]
    assert list(policy.gradient_study.path_steps) == [10, 50, 100]
    assert policy.replication17.adaptive_stability.enabled == (strategy != 'reuse')
    assert policy.replication17.adaptive_stability.get('snapshot_only', False) == (strategy == 'full')


def test_timing_tracks_failed_and_resumed_attempts(tmp_path):
    with pytest.raises(RuntimeError), rq4.training_timer(tmp_path, '1.7b', 'reuse', False):
        raise RuntimeError('interrupted')
    with rq4.training_timer(tmp_path, '1.7b', 'reuse', True):
        pass
    timing = rq4.read(tmp_path/'rq4_training_time.json')
    assert timing['complete_history'] and [r['status'] for r in timing['attempts']] == ['interrupted', 'complete']
    args = h200.parser().parse_args(['--scale', '1.7b', '--output', str(tmp_path), '--stage', 'eval'])
    assert h200.build_plan(args, {})['rq4_strategy'] == 'reuse'
