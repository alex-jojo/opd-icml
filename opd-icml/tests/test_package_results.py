"""Export actual data files, preserve their bytes, and never bundle model weights or keys."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import package_results as exports
import disk_space
import run_h200_experiment as h200
import run_h200_suite as suite


def save(root, relative, content=b'payload'):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_archive_preserves_all_experiment_data_and_excludes_models_and_credentials(tmp_path, monkeypatch):
    run = tmp_path / 'experiment output'
    data = {
        'eval/final_mean8.summary.json': b'{"mean_at_8": 0.75}',
        'eval/code_pass8/livecodebench/generations.jsonl': b'{"responses":["answer"]}\n',
        'training/training_records/1.jsonl': b'{"step":1}\n',
        'training/validation_records/20.jsonl': b'{"acc":1}\n',
        'training/replication/token_audit/step_001.npz': b'array-data-preserved-exactly',
        'training/assets/full_step10_rank128.npz': b'diagnostic-basis-data',
        'training/hydra/.hydra/config.yaml': b'trainer:\n  logger: [console, wandb]\n',
        'logs/train.log': b'train finished\n',
        'status.json': b'{"4b":"complete"}',
    }
    for relative, content in data.items():
        save(run, relative, content)
    excluded = [
        'training/checkpoints/global_step_106/actor/model_world_size_1_rank_0.pt',
        'merged/model.safetensors', 'models/base/model.bin',
        'merged/config.json', 'eval/stray_weights.pt', 'weights.npz', 'model.safetensors.index.json',
        'training/assets/model.onnx', 'adapter_model.safetensors', 'model.gguf',
        '.wandb.local.json', 'logs/.env', 'credentials.json', 'wandb/run/files/config.yaml',
        'older-results.zip', 'logs/settings.json',
    ]
    for relative in excluded:
        save(run, relative, b'private-or-model-file')
    monkeypatch.setenv('OPD_WANDB_CONFIG', str(run / 'logs/settings.json'))
    output = exports.package_results(run, phase='all')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        expected = {f'{run.name}/{name}' for name in data}
        assert set(archive.namelist()) == expected | {exports.MANIFEST, 'EXPORT_README.txt', 'READ_MOMENTS.py'}
        for relative, content in data.items():
            assert archive.read(f'{run.name}/{relative}') == content
        manifest = json.loads(archive.read(exports.MANIFEST))
        assert manifest['phase'] == 'all' and manifest['file_count'] == len(data)
        assert manifest['original_bytes'] == sum(map(len, data.values()))
        for entry in manifest['files']:
            assert entry['crc32'] == f'{archive.getinfo(entry["path"]).CRC:08x}'
        assert archive.getinfo(f'{run.name}/training/assets/full_step10_rank128.npz').compress_type == zipfile.ZIP_STORED


def test_regeneration_updates_results_without_including_its_own_zip(tmp_path):
    run = tmp_path / 'run'
    save(run, 'eval/summary.json', b'{"score":0.5}')
    output = run / 'exports/results.zip'
    exports.package_results(run, output)
    save(run, 'eval/summary.json', b'{"score":0.75}')
    exports.package_results(run, output, phase='eval')
    with zipfile.ZipFile(output) as archive:
        assert archive.read('run/eval/summary.json') == b'{"score":0.75}'
        assert not any(name.endswith('.zip') for name in archive.namelist())
        assert json.loads(archive.read(exports.MANIFEST))['file_count'] == 1


def test_running_out_of_space_while_packing_keeps_source_and_old_zip(tmp_path, monkeypatch):
    run = tmp_path/'run'
    raw = save(run, 'eval/base/responses.jsonl.gz', b'previous result')
    output = exports.package_results(run)
    previous = output.read_bytes()
    raw.write_bytes(b'x' * 5 * 1024**2)
    calls = []
    def available(path):
        calls.append(path)
        return SimpleNamespace(free=disk_space.GIB + (4 * 1024**2 if len(calls) < 3 else 1))
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', available)
    monkeypatch.setenv('OPD_MIN_FREE_GIB', '1')
    with pytest.raises(OSError, match='reserve'):
        exports.package_results(run)
    assert output.read_bytes() == previous and raw.stat().st_size == 5 * 1024**2
    assert not list(output.parent.glob('*.tmp'))


def test_packaging_failure_preserves_the_previous_complete_archive(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    save(run, 'eval/summary.json', b'first version')
    output = exports.package_results(run)
    previous = output.read_bytes()
    def fail_write(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(exports.zipfile.ZipFile, 'write', fail_write)
    with pytest.raises(OSError, match='disk full'):
        exports.package_results(run)
    assert output.read_bytes() == previous
    assert not list(output.parent.glob(output.name + '.*.tmp'))


def test_files_still_being_written_do_not_produce_a_completed_archive(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    source = save(run, 'logs/train.log', b'first')
    original_write = exports.zipfile.ZipFile.write
    def changing_write(archive, filename, *args, **kwargs):
        original_write(archive, filename, *args, **kwargs)
        source.write_bytes(b'changed while packaging')
    monkeypatch.setattr(exports.zipfile.ZipFile, 'write', changing_write)
    with pytest.raises(ValueError, match='still being written'):
        exports.package_results(run)
    assert not exports.archive_path(run).exists()


def test_unrelated_existing_zip_is_never_overwritten(tmp_path):
    run = tmp_path / 'run'
    save(run, 'eval/summary.json')
    output = tmp_path / 'existing.zip'
    output.write_bytes(b'unrelated original file')
    with pytest.raises(ValueError, match='already exists'):
        exports.package_results(run, output)
    assert output.read_bytes() == b'unrelated original file'


def test_a_directory_containing_only_models_cannot_produce_a_data_archive(tmp_path):
    run = tmp_path / 'run'
    save(run, 'checkpoints/model.pt')
    save(run, 'merged/model.safetensors')
    save(run, '.wandb.local.json', b'{"WANDB_API_KEY":"fake-private"}')
    with pytest.raises(ValueError, match='No experiment data files'):
        exports.package_results(run)
    assert not exports.archive_path(run).exists()


def test_symbolic_links_are_not_followed(tmp_path):
    run = tmp_path / 'run'
    save(run, 'eval/summary.json')
    outside = save(tmp_path, 'outside/secret.json', b'private')
    try:
        (run / 'eval/linked.json').symlink_to(outside)
        (run / 'linked-directory').symlink_to(outside.parent, target_is_directory=True)
    except OSError:
        pytest.skip('Creating symbolic links is unavailable on this Windows host')
    output = exports.package_results(run)
    with zipfile.ZipFile(output) as archive:
        assert 'run/eval/summary.json' in archive.namelist()
        assert not any('linked' in name or 'secret.json' in name for name in archive.namelist())


def test_manual_cli_packages_existing_results_without_loading_models(tmp_path):
    run = tmp_path / 'run'
    save(run, 'eval/summary.json', b'{"score":1}')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/package_results.py'), '--run-dir', str(run)],
                            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                            capture_output=True, text=True, encoding='utf-8', check=True)
    assert '[DOWNLOAD ZIP]' in result.stdout
    assert exports.archive_path(run).is_file()


@pytest.mark.parametrize('no_package', [False, True])
def test_single_experiment_packages_only_after_success(tmp_path, monkeypatch, capsys, no_package):
    run = tmp_path / 'run'
    arguments = ['run_h200_experiment.py', '--scale', 'deepseek-coder-1.3b',
                 '--stage', 'grade', '--output', str(run)]
    if no_package:
        arguments.append('--no-package')
    monkeypatch.setattr(sys, 'argv', arguments)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(h200, 'execute', lambda *args: save(run, 'eval/code_pass8/summary.json', b'{"score":1}'))
    h200.main()
    output = exports.archive_path(run, 'deepseek-coder-1.3b')
    assert output.exists() == (not no_package)
    assert ('[DOWNLOAD ZIP]' in capsys.readouterr().out) == (not no_package)


def test_failed_experiment_does_not_export_a_completed_results_zip(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['run_h200_experiment.py', '--scale', 'deepseek-coder-1.3b',
                                    '--stage', 'grade', '--output', str(run)])
    def fail(*args):
        raise subprocess.CalledProcessError(7, ['failed-judge'])
    monkeypatch.setattr(h200, 'execute', fail)
    monkeypatch.setattr(h200, 'package_results', lambda *args, **kwargs: pytest.fail('Failed stages must not export'))
    with pytest.raises(SystemExit) as error:
        h200.main()
    assert error.value.code == 7 and not exports.archive_path(run, 'deepseek-coder-1.3b').exists()


def test_dry_run_never_packages_even_when_results_already_exist(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'run'
    save(run, 'eval/summary.json', b'{"score":1}')
    monkeypatch.setattr(sys, 'argv', ['run_h200_experiment.py', '--scale', 'deepseek-coder-1.3b',
                                    '--stage', 'grade', '--dry-run', '--output', str(run)])
    monkeypatch.setattr(h200, 'execute', lambda *args: pytest.fail('Dry-run must not execute a stage'))
    monkeypatch.setattr(h200, 'package_results', lambda *args, **kwargs: pytest.fail('Dry-run must not write a ZIP'))
    h200.main()
    plan = json.loads(capsys.readouterr().out)
    assert not Path(plan['results_zip']).exists()


def test_suite_suppresses_individual_zips_and_exports_one_archive_after_all_jobs(tmp_path, monkeypatch):
    run = tmp_path / 'suite'
    locks = tmp_path / 'locks'
    locks.mkdir()
    monkeypatch.setattr(sys, 'argv', ['run_h200_suite.py', '--output', str(run)])
    monkeypatch.setattr(suite, 'os', SimpleNamespace(name='posix', environ={}))
    monkeypatch.setattr(suite, 'available_resources', lambda: (list(range(64)), 512))
    monkeypatch.setattr(suite.tempfile, 'gettempdir', lambda: str(locks))
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
    monkeypatch.setattr(suite, 'resolve_gpu_uuids', lambda *args: [
        f'{index:08x}-0000-0000-0000-000000000001' for index in range(4)])
    class FakeRunner:
        def run(self, *args, **kwargs):
            pass
        def cancel(self):
            pytest.fail('Successful suite must not cancel')
    monkeypatch.setattr(suite, 'Processes', FakeRunner)
    def finish(plan, runner):
        for job in [j for queue in plan['small_queues'] for j in queue] + plan['large']:
            assert '--no-package' in job['command']
            if job['grade']:
                assert '--no-package' in job['grade']
            if job['rq4_strategy']:
                summary = dict(scale=job['scale'], strategy=job['rq4_strategy'],
                               training_wall_seconds=60., five_benchmark_mean_at_8=.5)
                save(run, job['name']+'/training/training_records/rq4/summary.json', json.dumps(summary).encode())
                if job['shared_early_stop']:
                    shared = dict(summary, strategy='early-stop', source_strategy='full')
                    save(run, job['name']+'/training/training_records/rq4/early_stop_summary.json', json.dumps(shared).encode())
        save(run, 'status.json', b'{"__suite__":"complete"}')
        save(run, '4b/eval/final_mean8.summary.json', b'{"score":1}')
        save(run, '4b/merged/model.safetensors', b'never include')
    monkeypatch.setattr(suite, 'schedule', finish)
    suite.main()
    with zipfile.ZipFile(exports.archive_path(run)) as archive:
        assert archive.testzip() is None
        assert 'suite/4b/eval/final_mean8.summary.json' in archive.namelist()
        assert set(json.loads(archive.read('suite/rq4_comparison.json'))) == set(h200.EXPERIMENTS)
        assert json.loads(archive.read(exports.MANIFEST))['phase'] == 'suite-complete'
        assert not any('model.safetensors' in name for name in archive.namelist())
