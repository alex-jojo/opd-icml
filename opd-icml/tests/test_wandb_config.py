"""W&B credentials reach every launcher without entering commands or saved configs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import wandb_config
import run_h200_experiment as h200
import run_h200_suite as suite


@pytest.fixture
def private_config(tmp_path):
    path = tmp_path / 'private.json'
    path.write_text(json.dumps(dict(WANDB_API_KEY='test-secret-do-not-print', WANDB_MODE='online',
                                    WANDB_PROJECT='test-project')), encoding='utf-8')
    return path


def test_private_key_enables_tracking_and_explicit_environment_takes_precedence(private_config):
    env = wandb_config.load_wandb_environment({'OPD_WANDB_CONFIG': str(private_config)})
    assert env['WANDB_API_KEY'] == 'test-secret-do-not-print' and env['WANDB_MODE'] == 'online'
    config = dict(trainer=dict(logger=['console'], project_name='original'))
    overrides = wandb_config.apply_wandb_tracking(config, env)
    assert config['trainer']['logger'] == ['console', 'wandb']
    assert config['trainer']['project_name'] == 'test-project'
    assert 'WANDB_API_KEY' not in json.dumps(overrides)
    assert env['WANDB_API_KEY'] not in json.dumps(config)
    assert not wandb_config.apply_wandb_tracking(config, env), 'Applying tracking twice must not duplicate backends'
    env = wandb_config.load_wandb_environment(dict(OPD_WANDB_CONFIG=str(private_config),
        WANDB_API_KEY='environment-secret', WANDB_MODE='offline', WANDB_PROJECT='environment-project'))
    assert env['WANDB_API_KEY'] == 'environment-secret' and env['WANDB_MODE'] == 'offline'
    assert env['WANDB_PROJECT'] == 'environment-project'


def test_missing_default_config_preserves_offline_launches_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(wandb_config, 'PRIVATE_CONFIG', tmp_path / 'missing.json')
    env = wandb_config.load_wandb_environment({})
    assert env['WANDB_MODE'] == 'offline' and 'WANDB_API_KEY' not in env
    assert not wandb_config.apply_wandb_tracking(dict(trainer=dict(logger=['console'])), env)


def test_explicit_disable_keeps_console_only(private_config):
    env = wandb_config.load_wandb_environment(dict(OPD_WANDB_CONFIG=str(private_config), WANDB_MODE='disabled'))
    config = dict(trainer=dict(logger=['console']))
    assert not wandb_config.apply_wandb_tracking(config, env)
    assert config['trainer']['logger'] == ['console']


def test_h200_and_suite_children_load_private_credentials(private_config, tmp_path, monkeypatch):
    env = {'OPD_WANDB_CONFIG': str(private_config)}
    args = h200.parser().parse_args(['--scale', '4b'])
    child = h200.child_environment(h200.build_plan(args, env), env)
    assert child['WANDB_MODE'] == 'online' and child['WANDB_API_KEY'] == 'test-secret-do-not-print'
    monkeypatch.setenv('OPD_WANDB_CONFIG', str(private_config))
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.delenv('WANDB_MODE', raising=False)
    captured = []
    class Completed:
        pid = 4321
        def wait(self, **kwargs):
            return 0
    def popen(command, **kwargs):
        captured.append(kwargs['env'])
        return Completed()
    monkeypatch.setattr(suite.subprocess, 'Popen', popen)
    monkeypatch.setattr(suite, 'terminate_groups', lambda *args: None)
    suite.Processes().run(['dummy'], tmp_path / 'suite.log')
    assert captured[0]['WANDB_API_KEY'] == 'test-secret-do-not-print'
    assert captured[0]['WANDB_MODE'] == 'online'
    assert 'test-secret-do-not-print' not in (tmp_path / 'suite.log').read_text()


@pytest.mark.parametrize('scale', list(h200.EXPERIMENTS))
def test_all_seven_launch_plans_enable_wandb_without_printing_the_key(scale, private_config):
    env = os.environ.copy()
    env.update(OPD_WANDB_CONFIG=str(private_config), WANDB_MODE='online', CUDA_VISIBLE_DEVICES='0,1,2,3')
    env.pop('WANDB_API_KEY', None)
    env.pop('WANDB_PROJECT', None)
    command = [sys.executable, str(ROOT / 'scripts/run_h200_experiment.py'), '--scale', scale, '--dry-run']
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', check=True)
    plan = json.loads(result.stdout)
    assert plan['resolved_training']['wandb'] == dict(enabled=True, mode='online', project='test-project')
    actual = shlex.split(plan['resolved_training']['command'])
    logger = next(item for item in actual if item.startswith('++trainer.logger='))
    assert json.loads(logger.split('=', 1)[1]) == ['console', 'wandb']
    assert 'test-secret-do-not-print' not in result.stdout + result.stderr
    assert 'WANDB_API_KEY' not in result.stdout + result.stderr


def test_invalid_private_configuration_does_not_echo_values(tmp_path):
    path = tmp_path / 'invalid.json'
    path.write_text('{"WANDB_API_KEY": "private-value"', encoding='utf-8')
    with pytest.raises(ValueError) as error:
        wandb_config.load_wandb_environment({'OPD_WANDB_CONFIG': str(path)})
    assert 'private-value' not in str(error.value)
