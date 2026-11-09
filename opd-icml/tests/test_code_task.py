"""CPU data/protocol tests for Eurus code OPD; no external code is executed."""
import copy
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('prepare_eurus_code', ROOT / 'scripts/prepare_eurus_code.py')
judge = load('eurus_code_cpu_test', ROOT / 'runtime/verl/utils/reward_score/eurus_code.py')
metrics = load('opd_task_metrics_cpu_test', ROOT / 'runtime/verl/trainer/ppo/opd_task_metrics.py')
sys.path.insert(0, str(ROOT))
launcher = load('train_code_test', ROOT / 'train.py')


def row(ability='code', source='apps', fn_name=None):
    tests = dict(inputs=['1\n2', '3\n4'], outputs=['3', '7'])
    if fn_name:
        tests['fn_name'] = fn_name
    return dict(data_source=source, ability=ability,
                prompt=[dict(role='system', content='Reason carefully.'),
                        dict(role='user', content='Add the inputs. Write Python code in ```python fences.')],
                reward_model=dict(style='rule', ground_truth=json.dumps(tests)), extra_info=dict(index=10, split='dummy'))


def test_code_filter_provenance_and_tests_are_preserved(tmp_path):
    rows = [row('math', 'math_dapo'), row(source='apps'), row(source='codeforces', fn_name='add')]
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=1)
    original = source.read_bytes()
    output = tmp_path / 'code.parquet'
    prepare.prepare_split(source, output, 'train', expected_count=2)
    saved = pq.read_table(output).to_pylist()
    assert [r['data_source'] for r in saved] == ['apps', 'codeforces']
    assert all(r['ability'] == 'code' for r in saved)
    assert [r['extra_info']['source_index'] for r in saved] == [1, 2]
    assert [r['reward_model']['ground_truth'] for r in saved] == [r['reward_model']['ground_truth'] for r in rows[1:]]
    assert saved[0]['prompt'] == [dict(role='user', content='Reason carefully.\n\nAdd the inputs. Write Python code in ```python fences.')]
    assert '\\boxed' not in saved[0]['prompt'][0]['content']
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    assert manifest['call_based_problems'] == 1 and manifest['stdin_problems'] == 1
    assert manifest['ground_truth_preserved']
    assert manifest['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    before = output.read_bytes()
    prepare.prepare_split(source, output, 'train', expected_count=2)
    assert output.read_bytes() == before and source.read_bytes() == original
    launcher.validate_code_dataset(output)
    with pytest.raises(ValueError, match='code-only'):
        launcher.validate_code_dataset(source)


def test_preparation_refuses_incorrect_count_or_test_cases(tmp_path):
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist([row()]), source)
    output = tmp_path / 'code.parquet'
    with pytest.raises(ValueError, match='Expected 2'):
        prepare.prepare_split(source, output, 'train', expected_count=2)
    assert not output.exists()
    invalid = row()
    invalid['reward_model']['ground_truth'] = '{"inputs":[],"outputs":[]}'
    with pytest.raises(ValueError, match='Empty/misaligned'):
        prepare.convert_row(invalid, 'train', 0, 0)


def test_tampered_prepared_data_is_rejected(tmp_path):
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist([row()]), source)
    output = tmp_path / 'code.parquet'
    prepare.prepare_split(source, output, 'train', expected_count=1)
    with output.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(Exception):
        prepare.validate_existing(output, 'train', 1)


@pytest.mark.parametrize('statuses,expected', [([True, np.bool_(True)], 1.), ([True, False], 0.),
                                            ([True], 0.), ([-1, -1], 0.), ([-2], 0.)])
def test_submission_requires_all_tests_pass(monkeypatch, statuses, expected):
    calls = []
    def local(code, tests, timeout):
        calls.append((code, tests, timeout))
        return statuses
    monkeypatch.setattr(judge, '_local_results', local)
    spec = row()['reward_model']['ground_truth']
    score = judge.compute_score('apps', 'explanation\n```python\nprint(3)\n```', spec)
    assert score['acc'] == score['score'] == expected
    assert score['code_format'] == 1.
    assert score['code_test_count'] == 2
    assert calls[0][0] == 'print(3)' and calls[0][1] == json.loads(spec)


def test_batched_scores_preserve_order_and_handle_env_worker_string(monkeypatch):
    monkeypatch.setattr(judge, '_local_results', lambda code, tests, timeout: [True, True] if code == 'pass' else [False, False])
    scores = judge.compute_scores(['apps', 'taco'], ['pass', 'fail'],
                                  [row()['reward_model']['ground_truth']] * 2, workers='2')
    assert [s['acc'] for s in scores] == [1., 0.]
    with pytest.raises(ValueError, match='different lengths'):
        judge.compute_scores(['apps'], [], [])


def test_sandbox_failure_is_not_incorrect_solution(monkeypatch):
    name = 'verl.utils.reward_score.sandbox_fusion.utils'
    backend = types.ModuleType(name)
    backend.check_correctness = lambda **kwargs: ([-1, -1], [{'status': 'api_error'}] * 2)
    monkeypatch.setitem(sys.modules, name, backend)
    with pytest.raises(RuntimeError, match='service failed'):
        judge.compute_score('apps', 'print(3)', row()['reward_model']['ground_truth'],
                            backend='sandbox', sandbox_url='http://127.0.0.1/run_code')
    backend.check_correctness = lambda **kwargs: ([True, True], [{'status': 'success'}] * 2)
    assert judge.compute_score('apps', 'print(3)', row()['reward_model']['ground_truth'],
                                backend='sandbox', sandbox_url='http://127.0.0.1/run_code')['acc'] == 1.


def test_code_metrics_and_compact_test_reference():
    score = metrics.code_accuracy_metrics(.5, {'code_format': [.0, 1.], 'code_compile_error': [.0, .0]})
    assert score['monitor/train_code_accuracy'] == .5
    assert not any('boxed' in key or 'semantic' in key for key in score)
    spec = row()['reward_model']['ground_truth']
    reference = metrics.code_ground_truth_reference(spec)
    assert reference['test_count'] == 2 and reference['code_tests_sha256'] == hashlib.sha256(spec.encode()).hexdigest()
    assert 'inputs' not in reference and 'outputs' not in reference


def test_training_and_validation_dump_keeps_compact_test_references(tmp_path, monkeypatch):
    path = ROOT / 'runtime/verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RayPPOTrainer')
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == '_dump_generations')
    namespace = {'os': os, 'json': json}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec'), namespace)
    monkeypatch.setitem(sys.modules, 'verl.trainer.ppo.opd_task_metrics', metrics)
    dummy = types.SimpleNamespace(config=types.SimpleNamespace(data={'opd_task': 'code'}), global_steps=20)
    namespace['_dump_generations'](dummy, ['a prompt'], ['```python\nprint(3)\n```'],
                                   [row()['reward_model']['ground_truth']], [1.], {'acc': [1.]}, str(tmp_path))
    record = json.loads((tmp_path / '20.jsonl').read_text())
    assert record['score'] == 1. and record['gts']['test_count'] == 2
    assert 'inputs' not in record['gts'] and 'outputs' not in record['gts']


def test_code_launcher_passes_dataset_and_judge_to_runtime(tmp_path, monkeypatch):
    train, validation = tmp_path / 'train.parquet', tmp_path / 'validation.parquet'
    for path, split in [(train, 'train'), (validation, 'validation')]:
        converted = prepare.convert_row(row(), split, 0, 0)
        pq.write_table(pa.Table.from_pylist([converted], schema=prepare.SCHEMA), path)
    output = tmp_path / 'run'
    calls = []
    monkeypatch.setattr(launcher, 'prepare_models', lambda *args: {'student': '/prepared/student', 'teacher': '/prepared/teacher'})
    monkeypatch.setattr(launcher.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(sys, 'argv', ['train.py', '--task', 'code', '--scale', '4b', '--experiment', 'paired-full',
                                    '--data', str(train), '--val-data', str(validation), '--output', str(output),
                                    '--sandbox-url', 'http://127.0.0.1/run_code', '--code-workers', '3'])
    launcher.main()
    command, options = calls[0]
    assert command[command.index('--config-name') + 1] == 'code-paired-full'
    assert options['env']['OPD_DATA'] == str(train.resolve())
    assert options['env']['OPD_VALIDATION_DATA'] == str(validation.resolve())
    assert options['env']['OPD_CODE_JUDGE_BACKEND'] == 'sandbox'
    assert options['env']['OPD_CODE_WORKERS'] == '3'
    assert options['env']['OPD_STUDENT'] == '/prepared/student'


def test_code_launcher_rejects_math_data_before_preparing_models(tmp_path, monkeypatch):
    train, validation = tmp_path / 'train.parquet', tmp_path / 'validation.parquet'
    pq.write_table(pa.Table.from_pylist([row('math', 'math_dapo')]), train)
    pq.write_table(pa.Table.from_pylist([row()]), validation)
    def no_prepare(*args):
        raise AssertionError('Invalid datasets must fail before model preparation')
    monkeypatch.setattr(launcher, 'prepare_models', no_prepare)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--task', 'code', '--scale', '4b', '--experiment', 'paired-full',
                                    '--data', str(train), '--val-data', str(validation)])
    with pytest.raises(SystemExit):
        launcher.main()


@pytest.mark.parametrize('scale', ['1.7b', '4b', '8b', 'gemma3-12b'])
@pytest.mark.parametrize('experiment', ['paired-full', 'original-full', 'early-stop', 'reuse'])
def test_code_and_math_launchers(scale, experiment):
    for task in ['math', 'code']:
        result = subprocess.run([sys.executable, str(ROOT / 'train.py'), '--scale', scale,
                                 '--experiment', experiment, '--task', task, '--dry-run'],
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        assert report['task'] == task
        if task == 'code':
            assert 'runs/code' in report['output'].replace('\\', '/')
            assert report['data'].endswith('eurus_code_train.parquet')
            assert report['validation_data'].endswith('eurus_code_validation.parquet')
            assert report['max_prompt_tokens'] == 2048 and report['max_response_tokens'] == 3072
            assert '--config-name code-' + experiment in report['command']
        else:
            assert report['data'].endswith('dapo14k_train.parquet')


def test_real_hydra_composition_matches_launcher(monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    runtime_configs = ROOT / 'runtime/verl/trainer/config'
    for key, value in dict(OPD_RUN='/tmp/run', OPD_STUDENT='/tmp/student', OPD_TEACHER='/tmp/teacher',
                           OPD_DATA='/tmp/code_train.parquet', OPD_VALIDATION_DATA='/tmp/code_val.parquet',
                           OPD_PROJECT_ROOT=str(ROOT)).items():
        monkeypatch.setenv(key, value)
    for scale in ['1.7b', '4b', '8b', 'gemma3-12b']:
        for experiment in ['paired-full', 'original-full', 'early-stop', 'reuse']:
            with initialize_config_dir(config_dir=str(ROOT / 'configs' / scale), version_base=None):
                cfg = compose(config_name='code-' + experiment, overrides=[
                    'hydra.searchpath=' + json.dumps(['file://' + runtime_configs.as_posix()], ensure_ascii=False)])
            resolved = OmegaConf.to_container(cfg, resolve=True)
            _, expected = launcher.training_config(ROOT / 'configs' / scale, experiment, 'code')
            assert resolved['data']['opd_task'] == 'code'
            assert resolved['data']['val_files'] == ['/tmp/code_val.parquet']
            assert resolved['reward_model']['reward_manager'] == 'batch'
            assert resolved['custom_reward_function']['name'] == 'compute_scores'
            assert resolved['actor_rollout_ref']['actor']['policy_loss']['jump_reuse_updates'] == expected['actor_rollout_ref']['actor']['policy_loss']['jump_reuse_updates']
            assert resolved['trainer']['opd_pause_after_rollout_rounds'] == (53 if experiment == 'reuse' else 106)


def test_prime_module_fallback_runs_only_our_small_fixtures(monkeypatch):
    # Never execute dataset solutions here; these programs are owned test fixtures.
    module = load('prime_testing_util_cpu', ROOT / 'runtime/verl/utils/reward_score/prime_code/testing_util.py')
    monkeypatch.setattr(module, 'reliability_guard', lambda: None)
    monkeypatch.setattr(module.signal, 'alarm', lambda seconds: None, raising=False)
    tests = dict(inputs=['1\n2'], outputs=['3'])
    results, _ = module.run_test(copy.deepcopy(tests), 'a=int(input())\nb=int(input())\nprint(a+b)', debug=False)
    assert len(results) == 1 and all(bool(value == True) for value in results)
    function_tests = dict(inputs=['1\n2'], outputs=['3'], fn_name='add')
    results, _ = module.run_test(copy.deepcopy(function_tests), 'def add(a,b):\n    return a+b', debug=False)
    assert len(results) == 1 and all(bool(value == True) for value in results)
