"""Remove completed experiment model copies while preserving all measurements."""
import json
import os
from pathlib import Path
import shutil
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import project_absolute
from code_benchmarks import BENCHMARKS, EVALUATION_PROTOCOL, digest, save_json

TARGETS = ('training/checkpoints', 'merged')


def _linked(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, 'st_file_attributes', 0)
                                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0))


def _checked_files(run, relative):
    """Reject links/junctions and resolve every deletion target beneath this run."""
    target = run / relative
    current = target
    while current != run:
        if current.exists() or current.is_symlink():
            if _linked(current):
                raise ValueError(f'Refusing linked model path: {current}')
        current = current.parent
    if not target.exists():
        return []
    if not target.is_dir() or not target.resolve().is_relative_to(run):
        raise ValueError(f'Model cleanup target escapes its experiment: {target}')
    files = []
    for directory, subdirs, names in os.walk(target, followlinks=False):
        folder = Path(directory)
        for name in subdirs + names:
            path = folder / name
            if _linked(path) or not path.resolve().is_relative_to(run):
                raise ValueError(f'Refusing linked model artifact: {path}')
        files.extend(folder / name for name in names)
    return files


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _present(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Keep model files: missing evaluation artifact {path}')


def verify_completed_evaluation(run, task):
    """Check small completion metadata, without rehashing large response files."""
    if task == 'math':
        summary_path = run / 'eval/final_mean8.summary.json'
        _present(run / 'eval/final_mean8.jsonl')
        summary = _read(summary_path)
        if not summary or not all(
                row.get('problems', 0) > 0 and row.get('samples') == 8 * row['problems']
                and 0 <= row.get('mean_at_8', -1) <= 1 for row in summary.values()):
            raise ValueError('Keep model files: incomplete math evaluation summary')
        return [summary_path]
    if task != 'code':
        raise ValueError('Unknown evaluation task')
    root = run / 'eval/code_pass8'
    identity = _read(root / 'run.json')
    summary = _read(root / 'summary.json')
    if (identity.get('protocol') != EVALUATION_PROTOCOL or identity.get('sampling', {}).get('samples') != 8
            or set(identity.get('benchmarks', [])) != set(BENCHMARKS)
            or summary.get('metric') != 'pass@8' or set(summary.get('benchmarks', {})) != set(BENCHMARKS)):
        raise ValueError('Keep model files: incomplete code evaluation summary')
    proofs = [root / 'summary.json']
    for benchmark in BENCHMARKS:
        folder = root / benchmark
        report = _read(folder / 'summary.json')
        receipt = _read(folder / 'grading.complete.json')
        for name in ('generations.jsonl', 'grades.jsonl'):
            _present(folder / name)
        if (report != summary['benchmarks'][benchmark] or report.get('metric') != 'pass@8'
                or not 0 <= report.get('pass@8', -1) <= 1
                or report.get('problems', 0) <= 0 or report.get('samples') != 8 * report['problems']
                or report.get('samples_per_problem') != [8]
                or receipt.get('identity', {}).get('protocol') != EVALUATION_PROTOCOL
                or receipt.get('identity', {}).get('samples') != 8
                or receipt.get('files', {}).get('summary.json') != digest(folder / 'summary.json')):
            raise ValueError(f'Keep model files: incomplete {benchmark} grading receipt')
        proofs.append(folder / 'grading.complete.json')
    return proofs


def cleanup_completed_models(output, task):
    candidate = project_absolute(output)
    if _linked(candidate):
        raise ValueError('Refusing a linked experiment root')
    run = candidate.resolve()
    # Validate every target before deleting either one.
    inventories = {relative: _checked_files(run, relative) for relative in TARGETS}
    if not any((run / relative).exists() for relative in TARGETS):
        return
    proofs = verify_completed_evaluation(run, task)
    receipt_path = run / 'model_cleanup.json'
    metadata_path = run / 'model_artifact_metadata.json'
    metadata = _read(metadata_path) if metadata_path.exists() else {}
    records = []
    for relative, files in inventories.items():
        for path in files:
            name = path.relative_to(run).as_posix()
            size = path.stat().st_size
            records.append(dict(path=name, bytes=size))
            if path.suffix in {'.json', '.yaml', '.yml', '.txt'} and size <= 4 * 1024**2:
                metadata[name] = path.read_text(encoding='utf-8')
    # Keep checkpoint/config provenance even though weights are discarded.
    save_json(metadata_path, metadata)
    receipt = dict(protocol='completed-model-cleanup-v1', task=task, status='pending',
                   evaluation_proofs=[p.relative_to(run).as_posix() for p in proofs],
                   removed_directories=[], files=records, model_file_bytes=sum(r['bytes'] for r in records))
    save_json(receipt_path, receipt)
    for relative in TARGETS:
        target = run / relative
        # Revalidate immediately before removal. No shared cache is ever a target.
        _checked_files(run, relative)
        if target.exists():
            shutil.rmtree(target)
            receipt['removed_directories'].append(relative)
            save_json(receipt_path, receipt)
    receipt['status'] = 'complete'
    save_json(receipt_path, receipt)
    print(f'[CLEANUP] Removed checkpoint/merged copies: {receipt["model_file_bytes"] / 1024**3:.2f} GiB; '
          'kept measurements, responses, evaluation results and logs.', flush=True)


def finish_experiment(args, plan):
    if args.keep_model_files:
        return
    completed = args.stage in ('all', 'eval', 'grade') or (
        plan['task'] == 'math' and args.stage in ('train-generate', 'generate'))
    if completed:
        cleanup_completed_models(plan['output'], plan['task'])


def print_disk_space(output):
    directory = project_absolute(output)
    while not directory.exists():
        directory = directory.parent
    usage = shutil.disk_usage(directory)
    print(f'[DISK] Output filesystem: {usage.free / 1024**3:.1f} GiB free / '
          f'{usage.total / 1024**3:.1f} GiB total.', file=sys.stderr, flush=True)
