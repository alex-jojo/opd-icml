"""Deletion boundaries and completed-evaluation gating, using owned tiny fixtures."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import cleanup_model_artifacts as cleanup
import run_h200_experiment as h200
import run_h200_suite as suite
from code_benchmarks import BENCHMARKS, EVALUATION_PROTOCOL, digest, save_json


def write(root, name, data=b'data'):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def models(run):
    write(run, 'training/checkpoints/global_step_106/actor/model.pt', b'checkpoint')
    write(run, 'training/checkpoints/latest_checkpointed_iteration.txt', b'106')
    write(run, 'merged/model.safetensors', b'merged')
    write(run, 'merged/config.json', b'{"model_type":"test"}')


def math_results(run):
    write(run, 'eval/final_mean8.jsonl', b'{"acc_list":[1,0,0,0,0,0,0,0]}\n')
    save_json(run / 'eval/final_mean8.summary.json', {'math': dict(problems=1, samples=8, mean_at_8=.125)})


def code_results(run):
    root = run / 'eval/code_pass8'
    save_json(root / 'run.json', dict(protocol=EVALUATION_PROTOCOL, sampling={'samples': 8}, benchmarks=list(BENCHMARKS)))
    summaries = {}
    for name in BENCHMARKS:
        folder = root / name
        report = dict(metric='pass@8', **{'pass@8': 1.}, problems=1, samples=8, samples_per_problem=[8])
        save_json(folder / 'summary.json', report)
        write(folder, 'generations.jsonl')
        write(folder, 'grades.jsonl')
        save_json(folder / 'grading.complete.json', dict(identity=dict(protocol=EVALUATION_PROTOCOL, samples=8),
            files={'summary.json': digest(folder / 'summary.json')}))
        summaries[name] = report
    save_json(root / 'summary.json', dict(metric='pass@8', benchmarks=summaries))


@pytest.mark.parametrize('task', ['math', 'code'])
def test_cleanup_keeps_all_statistics_and_shared_model_cache(task, tmp_path):
    run = tmp_path / 'run'
    models(run)
    (math_results if task == 'math' else code_results)(run)
    retained = ['training/replication/results.npz', 'training/replication/fixed_bank.pt',
                'training/training_records/106.jsonl', 'logs/train.log']
    for path in retained:
        write(run, path)
    cache = write(tmp_path, 'models/shared.safetensors', b'shared')
    cleanup.cleanup_completed_models(run, task)
    assert all((run / path).read_bytes() == b'data' for path in retained)
    assert cache.read_bytes() == b'shared'
    assert not (run / 'merged').exists() and not (run / 'training/checkpoints').exists()
    receipt = json.loads((run / 'model_cleanup.json').read_text())
    assert receipt['status'] == 'complete' and receipt['model_file_bytes'] > 0
    metadata = json.loads((run / 'model_artifact_metadata.json').read_text())
    assert metadata['training/checkpoints/latest_checkpointed_iteration.txt'] == '106'
    before = (run / 'model_cleanup.json').read_bytes()
    cleanup.cleanup_completed_models(run, task)
    assert (run / 'model_cleanup.json').read_bytes() == before


@pytest.mark.parametrize('task', ['math', 'code'])
def test_incomplete_eval_never_deletes_model_files(tmp_path, task):
    run = tmp_path / 'run'
    models(run)
    (math_results if task == 'math' else code_results)(run)
    missing = ('eval/final_mean8.jsonl' if task == 'math' else
               'eval/code_pass8/humanevalplus/grading.complete.json')
    (run / missing).unlink()
    with pytest.raises((ValueError, OSError)):
        cleanup.cleanup_completed_models(run, task)
    assert (run / 'merged/model.safetensors').exists()
    assert (run / 'training/checkpoints/global_step_106/actor/model.pt').exists()


@pytest.mark.parametrize('location', ['merged', 'training', 'merged/nested', 'merged/link.bin'])
def test_linked_paths_are_rejected_before_any_deletion(tmp_path, location):
    run, outside = tmp_path / 'run', tmp_path / 'outside'
    outside.mkdir()
    write(outside, 'keep.txt', b'keep')
    run.mkdir()
    math_results(run)
    target = run / location
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if location.endswith('.bin'):
            target.symlink_to(outside / 'keep.txt')
        else:
            target.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Host cannot create symlinks')
    with pytest.raises(ValueError, match='linked'):
        cleanup.cleanup_completed_models(run, 'math')
    assert (outside / 'keep.txt').read_bytes() == b'keep'
    assert target.is_symlink()


@pytest.mark.parametrize('task,stage,expected', [
    ('math', 'all', True), ('math', 'eval', True), ('math', 'train-generate', True),
    ('math', 'generate', True), ('math', 'train', False),
    ('code', 'train-generate', False), ('code', 'generate', False), ('code', 'train', False),
    ('code', 'all', True), ('code', 'eval', True), ('code', 'grade', True)])
def test_cleanup_runs_only_after_complete_evaluation(task, stage, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(cleanup, 'cleanup_completed_models', lambda *args: calls.append(args))
    args = SimpleNamespace(stage=stage, keep_model_files=False)
    cleanup.finish_experiment(args, {'task': task, 'output': 'owned'})
    assert bool(calls) == expected
    calls.clear()
    args.keep_model_files = True
    cleanup.finish_experiment(args, {'task': task, 'output': 'owned'})
    assert not calls


def test_successful_grade_cleanup_precedes_zip(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    models(run)
    code_results(run)
    monkeypatch.setattr(sys, 'argv', ['run_h200_experiment.py', '--scale', 'deepseek-coder-1.3b',
                                    '--stage', 'grade', '--output', str(run)])
    monkeypatch.setattr(h200, 'execute', lambda *args: None)
    calls = []
    def package(*args, **kwargs):
        assert not (run / 'merged').exists() and not (run / 'training/checkpoints').exists()
        assert (run / 'model_cleanup.json').is_file()
        calls.append('packaged')
    monkeypatch.setattr(h200, 'package_results', package)
    h200.main()
    assert calls == ['packaged']


def test_failed_grade_keeps_models(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    models(run)
    monkeypatch.setattr(sys, 'argv', ['run_h200_experiment.py', '--scale', 'deepseek-coder-1.3b',
                                    '--stage', 'grade', '--output', str(run)])
    def fail(*args):
        raise subprocess.CalledProcessError(7, ['judge'])
    monkeypatch.setattr(h200, 'execute', fail)
    with pytest.raises(SystemExit):
        h200.main()
    assert (run / 'merged/model.safetensors').exists()
    assert (run / 'training/checkpoints/global_step_106/actor/model.pt').exists()


def test_suite_propagates_keep_option_to_generation_and_grading():
    args = suite.parser().parse_args(['--keep-model-files'])
    plan = suite.build_suite(args, {}, list(range(64)), 1024)
    for job in [j for q in plan['small_queues'] for j in q] + plan['large']:
        assert '--keep-model-files' in job['command']
        if job['grade']:
            assert '--keep-model-files' in job['grade']


@pytest.mark.parametrize('scale', list(h200.EXPERIMENTS))
def test_seven_profiles_save_only_final_model(scale):
    config = json.loads((ROOT / 'configs' / scale / 'paired-full.yaml').read_text())
    assert config['trainer']['save_freq'] == 106
    assert config['trainer']['max_actor_ckpt_to_keep'] == 1
    assert config['trainer']['opd_final_model_only_checkpoint']
    assert config['actor_rollout_ref']['actor']['checkpoint']['save_contents'] == ['model']
