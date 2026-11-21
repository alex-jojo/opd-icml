"""Exercise H200 launch plans, GPU isolation, and the train/merge/eval handoff on CPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_h200_experiment as h200

CASES = [
    ('deepseek_coder_1_3b', 'deepseek-coder-1.3b', 'code', 1),
    ('deepseek_coder_6_7b', 'deepseek-coder-6.7b', 'code', 1),
    ('deepseek_coder_33b', 'deepseek-coder-33b', 'code', 4),
    ('qwen3_1_7b', '1.7b', 'math', 1),
    ('qwen3_4b', '4b', 'math', 1),
    ('qwen3_8b', '8b', 'math', 4),
    ('gemma3_12b', 'gemma3-12b', 'math', 4),
]


def bash_binary():
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    return str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')


@pytest.mark.parametrize('name,scale,task,gpus', CASES)
def test_seven_actual_shell_entry_points_keep_train_and_evaluation_gpus_equal(name, scale, task, gpus, tmp_path):
    bash = bash_binary()
    if not bash:
        pytest.skip('Bash is unavailable')
    script = ROOT / 'scripts/h200' / f'train_eval_{name}.sh'
    subprocess.run([bash, '-n', script.as_posix()], check=True)
    env = os.environ.copy()
    env.update(PYTHON_BIN=Path(sys.executable).as_posix(), CUDA_VISIBLE_DEVICES='3,1,2,0')
    output = tmp_path / 'experiment output'
    result = subprocess.run([bash, script.as_posix(), '--dry-run', '--output', str(output)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['scale'] == scale and plan['task'] == task
    assert plan['training_gpus'] == plan['evaluation_gpus'] == plan['evaluation_tp'] == gpus
    assert plan['cuda_visible_devices'] == ('3' if gpus == 1 else '3,1,2,0')
    actual = plan['resolved_training']
    assert actual['gpu_count'] == actual['rollout_tensor_parallel_size'] == gpus
    assert actual['hardware_profile'] == f'h200x{gpus}'
    assert actual['rollout_batch'] == actual['mini_batch_size'] == 128
    assert actual['rounds_cap'] == 106 and actual['updates_per_round'] == 1
    assert actual['parameter_token_gradients']['sample_size'] == 64
    if task == 'code':
        assert plan['evaluate'][plan['evaluate'].index('--parallel-mode') + 1] == 'tensor'
        assert '--prompt-style' in plan['evaluate'] and 'chat' in plan['evaluate']
        assert plan['evaluate'][plan['evaluate'].index('--samples') + 1] == '8'
        assert 'eval/code_pass8' in next(v for v in plan['evaluate'] if 'eval/code_pass8' in v)
    else:
        assert plan['evaluate'][plan['evaluate'].index('--n') + 1] == '8'
    assert not output.exists(), 'dry-run must not create a run or download models/data'


@pytest.mark.parametrize('visible', ['', '-1', '0,0,1,2', '0,1', '0,,1,2'])
def test_four_card_experiments_fail_before_launch_with_invalid_masks(visible):
    with pytest.raises(ValueError, match='distinct visible GPUs'):
        h200.select_devices(4, None, {'CUDA_VISIBLE_DEVICES': visible})


def test_explicit_devices_preserve_order_and_cannot_override_card_rule():
    assert h200.select_devices(1, 'GPU-first', {}) == 'GPU-first'
    assert h200.select_devices(4, '3,2,1,0', {'CUDA_VISIBLE_DEVICES': '0'}) == '3,2,1,0'
    with pytest.raises(ValueError, match='exactly 1'):
        h200.select_devices(1, '0,1,2,3', {})


@pytest.mark.parametrize('scale,gpus', [('deepseek-coder-6.7b', 1), ('deepseek-coder-33b', 4)])
def test_new_deepseek_profiles_compose_and_pin_only_one_weight_format(scale, gpus):
    from hydra import compose, initialize_config_dir
    import train
    profile = ROOT / 'configs' / scale
    name, config = train.training_config(profile, 'paired-full', 'code')
    overrides = train.apply_hardware_profile(config, f'h200x{gpus}')
    search = 'hydra.searchpath=' + json.dumps(
        ['file://' + (ROOT / 'runtime/verl/trainer/config').as_posix()], ensure_ascii=False)
    with initialize_config_dir(config_dir=str(profile), version_base=None):
        cfg = compose(config_name=name, overrides=[search] + [
            '++' + key + '=' + json.dumps(value) for key, value in overrides.items()])
    assert cfg.trainer.n_gpus_per_node == cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == gpus
    assert cfg.actor_rollout_ref.actor.policy_loss.replication17.distributed_world_size == gpus
    assert cfg.data.opd_task == 'code' and cfg.data.max_response_length == 3072
    models = json.loads((profile / 'models.json').read_text())
    assert models['preparation'] == 'deepseek-coder-dual-eos-v2'
    for role, suffix in [('student', 'base'), ('teacher', 'instruct')]:
        spec = models[role]
        assert spec['repo'] == f'deepseek-ai/{scale}-{suffix}'
        assert len(spec['revision']) == 40 and int(spec['revision'], 16)
        assert '*.safetensors' in spec['allow_patterns'] and '*.bin' not in spec['allow_patterns']


def checkpoint_fixture(training, gpus, step=106):
    checkpoints = training / 'checkpoints'
    actor = checkpoints / f'global_step_{step}/actor'
    (actor / 'huggingface').mkdir(parents=True)
    (checkpoints / 'latest_checkpointed_iteration.txt').write_text(str(step))
    (actor / 'fsdp_config.json').write_text(json.dumps({'world_size': gpus}))
    (actor / 'huggingface/config.json').write_text('{}')
    for rank in range(gpus):
        (actor / f'model_world_size_{gpus}_rank_{rank}.pt').write_bytes(b'shard')
    return actor


@pytest.mark.parametrize('name,scale,task,gpus', CASES)
def test_pipeline_handoff_preserves_devices_and_can_retry_evaluation(name, scale, task, gpus, tmp_path, monkeypatch):
    import code_benchmarks
    monkeypatch.setattr(code_benchmarks, 'read_manifest', lambda *args: {})
    benchmark = tmp_path / 'math.jsonl'
    benchmark.write_text('{}\n')
    args = h200.parser().parse_args(['--scale', scale, '--output', str(tmp_path / 'run'), '--math-benchmarks', str(benchmark)])
    plan = h200.build_plan(args, {'CUDA_VISIBLE_DEVICES': '3,2,1,0'})
    plan['resolved_training'] = dict(rounds_cap=106, updates_per_round=1)
    env = h200.child_environment(plan, {'CUDA_VISIBLE_DEVICES': '3,2,1,0'})
    called = []

    def fake_run(command, environment, log):
        called.append(list(command))
        assert environment['CUDA_VISIBLE_DEVICES'] == ('' if '--grade-only' in command else ('3' if gpus == 1 else '3,2,1,0'))
        if command == plan['train']:
            actor = checkpoint_fixture(Path(plan['output']) / 'training', gpus)
            if plan['shared_early_stop']:
                import rq4_results as rq4
                run = h200.project_absolute(plan['output'])
                rq4.save_json(actor.parent/'opd_optimizer_updates.json', dict(rollout_round=106, optimizer_updates_done=106))
                for step in range(5, 106, 5):
                    rq4.save_json(run/f'training/training_records/rq4/early_stop_{step:03d}.json', dict(step=step, stable=False))
                rq4.save_json(actor.parent.parent/'early_stop/selection.json', dict(step=106, optimizer_updates=106,
                              interval=5, stop_reason='budget', uses_full_endpoint=True))
        if '--grade-only' in command:
            import rq4_results as rq4
            scores = {b: dict(problems=1, samples=8, **{'pass@8': 1.}) for b in rq4.CODE_BENCHMARKS}
            rq4.save_json(h200.project_absolute(plan['output'])/'eval/code_pass8/summary.json',
                          dict(metric='pass@8', benchmarks=scores))
        if 'verl.model_merger' in command:
            source = command[command.index('--local_dir') + 1]
            assert 'global_step_106' in source and '<latest>' not in source
            merged = Path(command[command.index('--target_dir') + 1])
            merged.mkdir()
            (merged / 'config.json').write_text('{}')
            (merged / 'model.safetensors').write_bytes(b'model')

    monkeypatch.setattr(h200, 'run_command', fake_run)
    h200.execute(args, plan, env)
    training_index = called.index(plan['train'])
    merge_index = next(i for i, command in enumerate(called) if 'verl.model_merger' in command)
    first_eval = plan['evaluate'] + (['--generate-only'] if task == 'code' else [])
    assert training_index < merge_index < called.index(first_eval)
    if task == 'code':
        assert called.index(first_eval) < called.index(plan['evaluate'] + ['--grade-only'])
    called.clear()
    args.stage = 'eval'
    h200.execute(args, plan, env)
    assert first_eval in called and plan['train'] not in called
    assert not any('verl.model_merger' in command for command in called)


def test_training_failure_never_merges_or_evaluates(tmp_path, monkeypatch):
    benchmark = tmp_path / 'math.jsonl'
    benchmark.write_text('{}\n')
    args = h200.parser().parse_args(['--scale', '8b', '--output', str(tmp_path / 'run'), '--math-benchmarks', str(benchmark)])
    plan = h200.build_plan(args, {})
    env = h200.child_environment(plan, {})
    called = []

    def fail_training(command, *unused):
        called.append(command)
        if command == plan['train']:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(h200, 'run_command', fail_training)
    with pytest.raises(subprocess.CalledProcessError):
        h200.execute(args, plan, env)
    assert plan['evaluate'] not in called
    assert not any('verl.model_merger' in command for command in called)


def test_checkpoint_rejects_partial_step_wrong_world_and_missing_rank(tmp_path):
    actor = checkpoint_fixture(tmp_path, 4)
    checkpoints = tmp_path / 'checkpoints'
    assert h200.final_checkpoint(checkpoints, 4, 106) == actor
    with pytest.raises(ValueError, match='final step'):
        h200.final_checkpoint(checkpoints, 4, 105)
    with pytest.raises(ValueError, match='GPU count'):
        h200.final_checkpoint(checkpoints, 1, 106)
    (actor / 'model_world_size_4_rank_3.pt').unlink()
    with pytest.raises(ValueError, match='Missing/empty'):
        h200.final_checkpoint(checkpoints, 4, 106)


def test_existing_training_output_is_never_overwritten(tmp_path, monkeypatch):
    args = h200.parser().parse_args(['--scale', '1.7b', '--output', str(tmp_path)])
    saved = tmp_path / 'keep.txt'
    saved.write_text('existing run')
    plan = h200.build_plan(args, {})
    monkeypatch.setattr(h200, 'run_command', lambda *args: pytest.fail('Must fail before running anything'))
    with pytest.raises(ValueError, match='empty --output'):
        h200.execute(args, plan, h200.child_environment(plan, {}))
    assert saved.read_text() == 'existing run'

