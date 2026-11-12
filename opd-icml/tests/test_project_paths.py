"""Paths survive a different launch directory and moving the whole project."""
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import project_paths as paths
import package_results as exports
import run_h200_experiment as h200
import run_h200_suite as suite
import evaluate_code
import wandb_config


def test_relative_inputs_use_project_root_even_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert paths.project_absolute('data/../data/train.parquet') == ROOT / 'data/train.parquet'
    assert paths.project_relative(ROOT / 'runs/test output') == 'runs/test output'
    assert paths.project_absolute('../shared/model') == ROOT.parent / 'shared/model'
    assert not paths.saved_path_matches('', ROOT)


@pytest.mark.parametrize('scale', h200.EXPERIMENTS)
def test_launch_plan_and_training_config_resolve_the_same_paths_from_any_cwd(scale, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_h200_experiment.py'),
        '--scale', scale, '--output', 'runs/path test', '--student', 'models/student',
        '--teacher', 'models/teacher', '--data', 'data/train.parquet', '--dry-run'],
        cwd=tmp_path, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3'},
        check=True, capture_output=True, text=True, encoding='utf-8')
    plan = json.loads(result.stdout)
    assert plan['path_base'] == plan['resolved_training']['path_base'] == 'project_root'
    assert plan['output'] == 'runs/path test'
    training = plan['resolved_training']
    assert training['output'] == training['command_cwd'] == 'runs/path test/training'
    assert training['data'] == 'data/train.parquet'
    command = shlex.split(training['command'])
    config_dir = command[command.index('--config-dir') + 1]
    assert not Path(config_dir).is_absolute()
    assert (ROOT / training['command_cwd'] / config_dir).resolve() == ROOT / 'configs' / scale
    assert 'hydra.run.dir=hydra' in command
    for command in [plan['train'], plan['merge'], plan['evaluate'], *plan['prepare_training']]:
        # The interpreter itself may be installed outside the project/on another drive.
        assert not any(Path(value).is_absolute() for value in command[1:])
    assert not (tmp_path / 'runs').exists()


def test_suite_children_share_project_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = suite.build_suite(suite.parser().parse_args(['--output', 'runs/suite1']),
                             {}, list(range(64)), 512)
    assert plan['output'] == 'runs/suite1' and plan['results_zip'] == 'runs/suite1-results.zip'
    jobs = [job for queue in plan['small_queues'] for job in queue] + plan['large']
    for job in jobs:
        assert job['output'] == 'runs/suite1/' + job['scale']
        for command in [job['command'], *([job['grade']] if job['grade'] else [])]:
            assert not any(Path(value).is_absolute() for value in command[1:])
    assert not any(Path(value).is_absolute() for command in plan['prepare'] for value in command[1:])


@pytest.mark.parametrize('legacy', [False, True])
def test_result_archive_can_be_updated_after_project_relocation(tmp_path, monkeypatch, legacy):
    old, new = tmp_path / 'old project', tmp_path / 'new project'
    run = old / 'runs/experiment1'
    run.mkdir(parents=True)
    (run / 'summary.json').write_text('{"score":0.5}')
    (run / 'model.safetensors').write_bytes(b'never export this')
    monkeypatch.setattr(paths, 'ROOT', old)
    output = exports.package_results('runs/experiment1')
    with zipfile.ZipFile(output) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries[exports.MANIFEST])
    assert manifest['source_directory'] == 'runs/experiment1'
    assert manifest['path_base'] == 'project_root'
    if legacy:
        manifest['source_directory'] = str(run)
        manifest.pop('path_base')
        entries[exports.MANIFEST] = json.dumps(manifest).encode()
        with zipfile.ZipFile(output, 'w') as archive:
            for name, content in entries.items():
                archive.writestr(name, content)
    shutil.copytree(old, new)
    monkeypatch.setattr(paths, 'ROOT', new)
    monkeypatch.chdir(tmp_path)
    (new / 'runs/experiment1/summary.json').write_text('{"score":1}')
    updated = exports.package_results('runs/experiment1')
    assert updated == new / 'runs/experiment1-results.zip'
    with zipfile.ZipFile(updated) as archive:
        assert archive.read('experiment1/summary.json') == b'{"score":1}'
        assert not any('safetensors' in name for name in archive.namelist())
        assert json.loads(archive.read(exports.MANIFEST))['source_directory'] == 'runs/experiment1'


def test_code_cache_relocation_still_checks_model_hashes_and_prompt_protocol():
    saved = dict(protocol='test', model=dict(path='/old/project/runs/merged', sha256={'model.safetensors': 'abc'}),
                 prompt_encoding='chat_template_token_ids_v1', sampling={'seed': 42})
    relocated = copy.deepcopy(saved)
    relocated['model']['path'] = 'runs/merged'
    assert evaluate_code.same_identity(saved, relocated)
    relocated['model']['sha256']['model.safetensors'] = 'changed'
    assert not evaluate_code.same_identity(saved, relocated)
    relocated = copy.deepcopy(saved)
    relocated.pop('prompt_encoding')
    assert not evaluate_code.same_identity(saved, relocated)


def test_wandb_environment_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = wandb_config.load_wandb_environment(
        {'WANDB_API_KEY': 'test-key', 'WANDB_PROJECT': 'portable-test'})
    assert env['WANDB_PROJECT'] == 'portable-test' and env['WANDB_MODE'] == 'online'
