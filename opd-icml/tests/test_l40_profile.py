"""CPU checks of the four-L40 launch overrides, with real Hydra composition."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('l40_launcher', ROOT / 'train.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('scale', ['1.7b', '4b', '8b', 'gemma3-12b'])
@pytest.mark.parametrize('task', ['math', 'code'])
def test_l40_preserves_protocol_and_matches_hydra(scale, task):
    from hydra import compose, initialize_config_dir
    profile = ROOT / 'configs' / scale
    name, cfg = launcher.training_config(profile, 'paired-full', task)
    original = copy.deepcopy(cfg)
    original_bytes = (profile / (name + '.yaml')).read_bytes()
    updates = launcher.apply_hardware_profile(cfg, 'l40x4')
    assert cfg['trainer']['n_gpus_per_node'] == 4
    assert cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size'] == 4
    assert cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'] == 1
    assert cfg['actor_rollout_ref']['actor']['policy_loss']['replication17']['distributed_world_size'] == 4
    assert cfg['actor_rollout_ref']['actor']['policy_loss'] == {
        **original['actor_rollout_ref']['actor']['policy_loss'],
        'replication17': {**original['actor_rollout_ref']['actor']['policy_loss']['replication17'], 'distributed_world_size': 4}}
    assert cfg['data'] == original['data']
    assert cfg['actor_rollout_ref']['actor']['ppo_mini_batch_size'] == original['actor_rollout_ref']['actor']['ppo_mini_batch_size']
    assert cfg['actor_rollout_ref']['actor']['policy_loss']['gradient_study']['parameter_tokens']['enabled']
    assert (profile / (name + '.yaml')).read_bytes() == original_bytes
    search = 'hydra.searchpath=' + json.dumps(['file://' + (ROOT / 'runtime/verl/trainer/config').as_posix()], ensure_ascii=False)
    with initialize_config_dir(config_dir=str(profile), version_base=None):
        composed = compose(config_name=name, overrides=[search] + ['++' + k + '=' + json.dumps(v) for k, v in updates.items()])
    for key, value in updates.items():
        target = composed
        for part in key.split('.'):
            target = target[part]
        assert target == value


def test_launch_command_and_explicit_gradient_off():
    result = subprocess.run([sys.executable, str(ROOT / 'train.py'), '--scale', '4b', '--task', 'code',
        '--experiment', 'paired-full', '--hardware', 'l40x4', '--parameter-token-gradients', 'off', '--dry-run'],
        check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report['gpu_count'] == 4 and report['hardware_profile'] == 'l40x4'
    assert report['micro_batch_size_per_gpu'] == 1 and report['rollout_tensor_parallel_size'] == 4
    assert not report['parameter_token_gradients']['enabled']
    assert '++trainer.n_gpus_per_node=4' in report['command']
    assert 'gradient_study.parameter_tokens.enabled=false' in report['command']


def test_single_gpu_probes_are_not_silently_removed():
    _, cfg = launcher.training_config(ROOT / 'configs/1.7b', 'original-full', 'math')
    original = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match='single-GPU disposable shadow probes'):
        launcher.apply_hardware_profile(cfg, 'l40x4')
    assert cfg == original


def test_default_keeps_recorded_hardware():
    _, cfg = launcher.training_config(ROOT / 'configs/1.7b', 'paired-full', 'math')
    original = copy.deepcopy(cfg)
    assert launcher.apply_hardware_profile(cfg, None) == {}
    assert cfg == original
