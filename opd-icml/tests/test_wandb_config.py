"""W&B command-line settings propagate without appearing in plans or logs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import wandb_config
import run_h200_experiment as h200
import run_h200_suite as suite


def test_environment_key_enables_tracking_without_persisting_credentials():
    source = dict(WANDB_API_KEY='test-secret-do-not-print', WANDB_PROJECT='test-project')
    env = wandb_config.load_wandb_environment(source)
    assert 'WANDB_MODE' not in source
    assert env['WANDB_MODE'] == 'online'
    config = dict(trainer=dict(logger=['console'], project_name='original'))
    overrides = wandb_config.apply_wandb_tracking(config, env)
    assert config['trainer']['logger'] == ['console', 'wandb']
    assert config['trainer']['project_name'] == 'test-project'
    assert env['WANDB_API_KEY'] not in json.dumps([config, overrides])
    assert 'WANDB_API_KEY' not in json.dumps(overrides)
    assert not wandb_config.apply_wandb_tracking(config, env)


def test_no_credentials_defaults_to_console_without_reading_files():
    env = wandb_config.load_wandb_environment({'OPD_WANDB_CONFIG': 'missing-legacy-file.json'})
    assert env['WANDB_MODE'] == 'disabled'
    assert not wandb_config.apply_wandb_tracking(dict(trainer=dict(logger=['console'])), env)


@pytest.mark.parametrize('with_key', [False, True])
def test_explicit_offline_enables_local_tracking(with_key):
    env = wandb_config.load_wandb_environment(dict(WANDB_MODE='offline',
        **({'WANDB_API_KEY': 'test-secret'} if with_key else {})))
    config = dict(trainer=dict(logger=['console']))
    wandb_config.apply_wandb_tracking(config, env)
    assert env['WANDB_MODE'] == 'offline'
    assert config['trainer'] == dict(logger=['console', 'wandb'], project_name='opd-icml')


@pytest.mark.parametrize('logger', [['console'], ['console', 'wandb'], ['tracking'], 'wandb'])
def test_explicit_disable_keeps_console_only(logger):
    env = wandb_config.load_wandb_environment(dict(WANDB_API_KEY='test-secret', WANDB_MODE='disabled'))
    config = dict(trainer=dict(logger=logger))
    wandb_config.apply_wandb_tracking(config, env)
    assert config['trainer']['logger'] == ['console']


def test_h200_and_suite_children_inherit_environment(tmp_path, monkeypatch):
    env = dict(WANDB_API_KEY='test-secret-do-not-print', WANDB_ENTITY='test-team')
    args = h200.parser().parse_args(['--scale', '4b'])
    child = h200.child_environment(h200.build_plan(args, env), env)
    assert child['WANDB_MODE'] == 'online'
    assert child['WANDB_API_KEY'] == env['WANDB_API_KEY']
    assert child['WANDB_ENTITY'] == 'test-team'
    monkeypatch.setenv('WANDB_API_KEY', env['WANDB_API_KEY'])
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
    assert captured[0]['WANDB_API_KEY'] == env['WANDB_API_KEY']
    assert captured[0]['WANDB_MODE'] == 'online'
    assert env['WANDB_API_KEY'] not in (tmp_path / 'suite.log').read_text()


@pytest.mark.parametrize('scale', list(h200.EXPERIMENTS))
def test_all_seven_launch_plans_enable_wandb_without_printing_key(scale):
    env = os.environ.copy()
    env.update(WANDB_API_KEY='test-secret-do-not-print', WANDB_MODE='online',
               WANDB_PROJECT='test-project', CUDA_VISIBLE_DEVICES='0,1,2,3')
    command = [sys.executable, str(ROOT / 'scripts/run_h200_experiment.py'),
               '--scale', scale, '--dry-run']
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, encoding='utf-8', check=True)
    plan = json.loads(result.stdout)
    assert plan['resolved_training']['wandb'] == dict(enabled=True, mode='online', project='test-project')
    actual = shlex.split(plan['resolved_training']['command'])
    logger = next(item for item in actual if item.startswith('++trainer.logger='))
    assert json.loads(logger.split('=', 1)[1]) == ['console', 'wandb']
    assert 'test-secret-do-not-print' not in result.stdout + result.stderr
    assert 'WANDB_API_KEY' not in result.stdout + result.stderr


def test_invalid_mode_does_not_echo_values():
    with pytest.raises(ValueError) as error:
        wandb_config.load_wandb_environment(dict(WANDB_MODE='private-value'))
    assert 'private-value' not in str(error.value)
