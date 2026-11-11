"""Create a portable experiment-data ZIP, excluding all model and credential files."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import project_absolute, project_relative, saved_path_matches

PROTOCOL = 'opd-experiment-data-zip-v1'
MANIFEST = 'EXPORT_MANIFEST.json'
EXCLUDED_DIRS = {
    'checkpoints', 'checkpoint', 'merged', 'models', 'model', 'huggingface',
    '.git', '.venv', '__pycache__', '.pytest_cache', 'wandb',
}
MODEL_SUFFIXES = {
    '.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.onnx', '.gguf', '.ggml',
    '.model', '.h5', '.hdf5', '.pkl', '.pickle', '.joblib', '.tflite',
}
DATA_SUFFIXES = {
    '.json', '.jsonl', '.npz', '.npy', '.csv', '.tsv', '.parquet', '.txt',
    '.log', '.md', '.yaml', '.yml', '.pdf', '.svg', '.png', '.jpg', '.jpeg', '.html',
}
CREDENTIAL_NAMES = {'.wandb.local.json', '.netrc', '_netrc', 'credentials.json'}


def archive_path(run, label=None):
    run = project_absolute(run)
    prefix = f'{label}-' if label else ''
    return run.with_name(f'{prefix}{run.name}-results.zip')


def _data_file(path):
    name = path.name.lower()
    if name in CREDENTIAL_NAMES or name == '.env' or name.startswith('.env.'):
        return False
    if path.suffix.lower() in MODEL_SUFFIXES:
        return False
    if name.endswith(('.safetensors.index.json', '.bin.index.json')):
        return False
    if name in {'model.npz', 'weights.npz', 'pytorch_model.npz', 'adapter_model.npz'}:
        return False
    return (path.suffix.lower() in DATA_SUFFIXES or
            name.endswith(('.jsonl.gz', '.json.gz', '.csv.gz', '.tsv.gz', '.txt.gz')))


def result_files(run):
    """Do not follow links or collect files from model/cache directories."""
    private_config = project_absolute(os.environ.get('OPD_WANDB_CONFIG', '.wandb.local.json')).resolve()
    for directory, subdirs, names in os.walk(run, followlinks=False):
        folder = Path(directory)
        subdirs[:] = sorted(name for name in subdirs
            if name.lower() not in EXCLUDED_DIRS and not (folder / name).is_symlink()
            and (folder / name).resolve().is_relative_to(run))
        for name in sorted(names):
            path = folder / name
            if (not path.is_symlink() and path.is_file() and _data_file(path)
                    and path.resolve().is_relative_to(run) and path.resolve() != private_config):
                yield path


def package_results(run, output=None, *, phase='manual'):
    run = project_absolute(run).resolve()
    if not run.is_dir():
        raise ValueError(f'Experiment directory does not exist: {run}')
    output = project_absolute(output).resolve() if output else archive_path(run)
    if output.suffix.lower() != '.zip':
        raise ValueError('Results archive output must end in .zip')
    files = list(result_files(run))
    if not files:
        raise ValueError(f'No experiment data files to package in {run}')
    if output.exists():
        try:
            with zipfile.ZipFile(output) as previous:
                saved = json.loads(previous.read(MANIFEST))
            if (saved.get('protocol') != PROTOCOL
                    or not saved_path_matches(saved.get('source_directory', ''), run)):
                raise ValueError
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            raise ValueError(f'Output already exists and is not this experiment data archive: {output}') from None
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    entries = []
    print(f'[ZIP] Packaging {len(files)} experiment data files from {project_relative(run)}', flush=True)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=3, allowZip64=True, strict_timestamps=False) as archive:
            for path in files:
                if path.is_symlink() or not path.resolve().is_relative_to(run):
                    raise ValueError(f'Experiment file changed into a link during packaging: {path}')
                before = path.stat()
                relative = path.relative_to(run).as_posix()
                member = f'{run.name}/{relative}'
                # NPZ and gzip data are already compressed; store them directly.
                compression = zipfile.ZIP_STORED if path.suffix.lower() in {'.npz', '.gz', '.png', '.jpg', '.jpeg', '.pdf'} else zipfile.ZIP_DEFLATED
                archive.write(path, member, compress_type=compression, compresslevel=3)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f'Experiment file is still being written; retry after it finishes: {path}')
                info = archive.getinfo(member)
                entries.append(dict(path=member, bytes=info.file_size, crc32=f'{info.CRC:08x}'))
            manifest = dict(protocol=PROTOCOL, source_directory=project_relative(run), path_base='project_root', phase=phase,
                created_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                contents='experiment data only; no models, checkpoints, credentials or nested archives',
                file_count=len(entries), original_bytes=sum(item['bytes'] for item in entries), files=entries,
                excluded_directories=sorted(EXCLUDED_DIRS), excluded_model_suffixes=sorted(MODEL_SUFFIXES))
            archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
            archive.writestr('EXPORT_README.txt',
                'Experiment data export\n'
                f'Phase: {phase}\n'
                'The source folder layout is preserved. Evaluation results, answers, scores,\n'
                'training/validation records, diagnostic arrays, configs and logs are included.\n'
                'Models, checkpoint weights, credentials, links and nested archives are excluded.\n'
                'See EXPORT_MANIFEST.json for the exact file list, sizes and ZIP CRC32 values.\n'
                'Exports from train/generate phases may not yet include final evaluation scores.\n')
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'[DOWNLOAD ZIP] {project_relative(output)} ({len(entries)} data files, {output.stat().st_size / 1024**2:.2f} MiB)', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='One experiment directory or an entire H200 suite')
    parser.add_argument('--output', type=Path, help='ZIP path; defaults to <run-dir>-results.zip beside the source directory')
    args = parser.parse_args()
    try:
        package_results(args.run_dir, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
