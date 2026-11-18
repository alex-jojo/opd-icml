import json
import shlex
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
from package_results import package_results
from test_h200_experiments import checkpoint_fixture


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
    plan = suite.build_suite(suite.parser().parse_args(['--output', str(tmp_path/'suite'), '--scales', '1.7b', '4b']),
                             {}, list(range(64)), 512)
    paths = []
    base_runs = []
    for queue in plan['small_queues']:
        assert [job['rq4_strategy'] for job in queue] == list(rq4.STRATEGIES)
        for job in queue:
            args = h200.parser().parse_args(job['command'][2:])
            strategy = args.rq4_strategy
            run = h200.project_absolute(job['output'])
            q = h200.build_plan(args, {})
            cap, split, final = (53, 2, 53) if strategy == 'reuse' else (106, 1, 40 if strategy == 'early-stop' else 106)
            q['resolved_training'] = dict(rounds_cap=cap, updates_per_round=split)
            window = run/'training/training_records/rq4'

            def launch(command, env, log):
                if command == q['train']:
                    assert not (run/'training').exists()
                    actor = checkpoint_fixture(run/'training', 1, final)
                    rq4.save_json(actor.parent/'opd_optimizer_updates.json', dict(rollout_round=final, optimizer_updates_done=final*split))
                    if strategy == 'early-stop':
                        rq4.save_json(window/'early_stop_040.json', dict(step=40, stable=True))
                elif 'verl.model_merger' in command:
                    assert f'global_step_{final}/actor' in command[command.index('--local_dir')+1]
                    merged = h200.project_absolute(q['merged_dir'])
                    merged.mkdir(parents=True)
                    (merged/'config.json').write_text('{}')
                    (merged/'weights.safetensors').touch()
                elif command == q['evaluate']:
                    values = dict(zip(rq4.BENCHMARKS, (.5, .2, .1, .3, .4)))
                    values['amc23'] = 1.
                    rq4.save_json(run/'eval/final_mean8.summary.json', {
                        k: dict(problems=i+1, samples=8*(i+1), mean_at_8=v) for i,(k,v) in enumerate(values.items())})
                elif command == q['evaluate_base']:
                    assert '--no-package' in command
                    base_runs.append((args.scale, strategy))
                    scores = dict(mean_at_8=dict.fromkeys(rq4.BENCHMARKS, .1), five_benchmark_mean_at_8=.1)
                    rq4.save_json(run/'eval/base/summary.json', dict(scale='4b', strategy='base', model=q['models']['student'],
                                  rollout_rounds=0, optimizer_updates=0, training_wall_seconds=None, strict_last_box=scores))
            monkeypatch.setattr(h200, 'run_command', launch)
            h200.execute(args, q, {})
            saved = rq4.read(window/'summary.json')
            assert saved['optimizer_updates'] == final*split and saved['rollout_rounds'] == final
            assert saved['stop_reason'] == ('stable' if strategy == 'early-stop' else 'budget')
            assert saved['five_benchmark_mean_at_8'] == pytest.approx(.3)
            assert saved['training_wall_seconds'] > 0
            paths.append(run)
    assert len(set(paths)) == 6
    assert base_runs == [('4b', 'full')]
    rq4.save_comparison(plan)
    root = h200.project_absolute(plan['output'])
    combined = rq4.read(root/'rq4_comparison.json')
    for arms in combined.values():
        for strategy in rq4.STRATEGIES:
            row = arms[strategy]
            assert row['training_time_saved_fraction'] == pytest.approx(1-row['training_wall_seconds']/arms['full']['training_wall_seconds'])
    baseline = combined['4b']['base']
    assert baseline['five_benchmark_change_pp'] == pytest.approx(-20)
    assert baseline['scoring_mode'] == 'strict_last_box' and baseline['training_time_saved_fraction'] is None
    assert 'base' not in combined['1.7b']
    with ZipFile(package_results(root)) as archive:
        assert 'suite/rq4_comparison.json' in archive.namelist()
        assert sum(n.endswith('/rq4/summary.json') for n in archive.namelist()) == 6
        assert sum(n.endswith('/eval/base/summary.json') for n in archive.namelist()) == 1


def test_early_stop_rejects_unconfirmed_partial_checkpoint(tmp_path):
    actor = checkpoint_fixture(tmp_path, 1, 20)
    rq4.save_json(actor.parent/'opd_optimizer_updates.json', dict(rollout_round=20, optimizer_updates_done=20))
    with pytest.raises(ValueError, match='confirmed stability stop'):
        rq4.endpoint(tmp_path/'checkpoints', tmp_path, 'early-stop', 106, 1)


@pytest.mark.parametrize('scale', rq4.SCALES)
@pytest.mark.parametrize('strategy', rq4.STRATEGIES)
def test_matched_real_hydra_config(scale, strategy):
    from hydra import compose, initialize_config_dir
    args = h200.parser().parse_args(['--scale', scale, '--rq4-strategy', strategy])
    cmd = h200.build_plan(args, {})['train']
    dry = json.loads(subprocess.check_output(cmd+['--dry-run'], cwd=ROOT, encoding='utf-8'))
    overrides = [s for s in shlex.split(dry['command']) if s.startswith('++')]
    search = 'hydra.searchpath='+json.dumps(['file://'+(ROOT/'runtime/verl/trainer/config').as_posix()], ensure_ascii=False)
    with initialize_config_dir(config_dir=str(ROOT/'configs'/scale), version_base=None):
        cfg = compose(config_name='paired-full', overrides=[search, *overrides])
    policy = cfg.actor_rollout_ref.actor.policy_loss
    assert policy.jump_reuse_updates == policy.replication17.reuse == (2 if strategy == 'reuse' else 1)
    assert policy.replication17.rounds == cfg.trainer.opd_pause_after_rollout_rounds == cfg.trainer.total_training_steps
    assert cfg.trainer.total_training_steps*policy.jump_reuse_updates == 106
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert policy.gradient_study.parameter_tokens.sample_size == 256 and policy.fixed_heldout_probe_size == 0
    assert policy.replication17.adaptive_stability.enabled == (strategy == 'early-stop')


def test_timing_tracks_failed_and_resumed_attempts(tmp_path):
    with pytest.raises(RuntimeError), rq4.training_timer(tmp_path, '1.7b', 'reuse', False):
        raise RuntimeError('interrupted')
    with rq4.training_timer(tmp_path, '1.7b', 'reuse', True):
        pass
    timing = rq4.read(tmp_path/'rq4_training_time.json')
    assert timing['complete_history'] and [r['status'] for r in timing['attempts']] == ['interrupted', 'complete']
    args = h200.parser().parse_args(['--scale', '1.7b', '--output', str(tmp_path), '--stage', 'eval'])
    assert h200.build_plan(args, {})['rq4_strategy'] == 'reuse'
