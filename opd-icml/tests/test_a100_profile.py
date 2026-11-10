"""Single-A100 overrides must preserve the OPD experiment and diagnostics."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('a100_launcher', ROOT / 'train.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('experiment', ['paired-full', 'original-full', 'early-stop', 'reuse'])
def test_single_a100_preserves_experiment_and_composes_hydra(experiment):
    from hydra import compose, initialize_config_dir
    profile = ROOT / 'configs/deepseek-coder-1.3b'
    name, cfg = launcher.training_config(profile, experiment, 'code')
    original = copy.deepcopy(cfg)
    recorded = {path: path.read_bytes() for path in profile.glob('*.yaml')}
    overrides = launcher.apply_hardware_profile(cfg, 'a100x1')
    assert cfg['trainer']['n_gpus_per_node'] == 1
    assert cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size'] == 1
    assert cfg['actor_rollout_ref']['rollout']['max_num_seqs'] == 16
    assert cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'] == 1
    assert cfg['actor_rollout_ref']['model']['enable_gradient_checkpointing']
    assert cfg['actor_rollout_ref']['actor']['fsdp_config']['optimizer_offload']
    assert cfg['actor_rollout_ref']['ref']['fsdp_config']['param_offload']
    assert cfg['actor_rollout_ref']['actor']['policy_loss'] == {
        **original['actor_rollout_ref']['actor']['policy_loss'],
        'replication17': {
            **original['actor_rollout_ref']['actor']['policy_loss']['replication17'],
            'distributed_world_size': 1,
        },
    }
    assert cfg['data'] == original['data']
    assert cfg['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == original['actor_rollout_ref']['actor']['ppo_mini_batch_size']
    assert cfg['trainer']['opd_pause_after_rollout_rounds'] == (53 if experiment == 'reuse' else 106)
    if experiment == 'paired-full':
        assert cfg['data']['train_batch_size'] == 128
        assert cfg['actor_rollout_ref']['actor']['policy_loss']['gradient_study']['parameter_tokens'] == {
            'enabled': True, 'mode': 'kl_stratified', 'sample_size': 256, 'seed': 42}
    search = 'hydra.searchpath=' + json.dumps(['file://' + (ROOT / 'runtime/verl/trainer/config').as_posix()], ensure_ascii=False)
    with initialize_config_dir(config_dir=str(profile), version_base=None):
        composed = compose(config_name=name, overrides=[search] + [
            '++' + key + '=' + json.dumps(value) for key, value in overrides.items()])
    for key, value in overrides.items():
        target = composed
        for part in key.split('.'):
            target = target[part]
        assert target == value
    assert all(path.read_bytes() == content for path, content in recorded.items())


def test_single_gpu_shadow_probes_are_preserved():
    _, cfg = launcher.training_config(ROOT / 'configs/1.7b', 'original-full', 'math')
    original = copy.deepcopy(cfg['actor_rollout_ref']['actor']['policy_loss']['replication17'])
    assert original['heavy_steps']
    launcher.apply_hardware_profile(cfg, 'a100x1')
    replication = cfg['actor_rollout_ref']['actor']['policy_loss']['replication17']
    assert replication == {**original, 'distributed_world_size': 1}


def test_cli_launches_one_gpu_with_exact_parameter_diagnostics():
    result = subprocess.run([
        sys.executable, str(ROOT / 'train.py'), '--scale', 'deepseek-coder-1.3b', '--task', 'code',
        '--experiment', 'paired-full', '--hardware', 'a100x1',
        '--parameter-token-gradients', 'exact-all', '--dry-run',
    ], check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report['hardware_profile'] == 'a100x1' and report['gpu_count'] == 1
    assert report['rollout_tensor_parallel_size'] == 1
    assert report['parameter_token_gradients'] == {'enabled': True, 'mode': 'exact_all'}
    assert report['rollout_batch'] == 128 and report['rounds_cap'] == 106
    assert '++trainer.n_gpus_per_node=1' in report['command']
    assert '++actor_rollout_ref.actor.policy_loss.replication17.distributed_world_size=1' in report['command']
    assert '++actor_rollout_ref.rollout.tensor_model_parallel_size=1' in report['command']
