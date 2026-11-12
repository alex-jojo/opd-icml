"""Prepare pinned LiveCodeBench and HumanEval+ evaluation data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

from code_benchmarks import BENCHMARKS, PROTOCOL, ROOT, date_selected, digest, lcb_sample, lockfile, read_jsonl, read_manifest, save_json, save_jsonl
sys.path.insert(0, str(ROOT))
from project_paths import project_absolute


def source_file(spec, cache, relative, sha256, source_dir=None):
    if source_dir:
        path = Path(source_dir) / relative
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(spec['repo'], relative, repo_type='dataset', revision=spec['revision'], cache_dir=str(cache)))
    if digest(path) != sha256:
        raise ValueError(f'Source checksum mismatch: {path}')
    return path


def official_release(spec, cache, source_dir=None):
    name = spec['official_url'].rsplit('/', 1)[-1]
    path = Path(source_dir) / name if source_dir else cache / name
    if not path.exists():
        if source_dir:
            raise FileNotFoundError(path)
        cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(name + '.tmp')
        with urllib.request.urlopen(spec['official_url'], timeout=60) as response, temporary.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(path)
    if digest(path) != spec['official_sha256']:
        raise ValueError(f'Official EvalPlus test checksum mismatch: {path}')
    return path


def build_evalplus(benchmark, spec, hf_path, official_path, directory):
    if benchmark != 'humanevalplus':
        raise ValueError('Only HumanEval+ is supported by the EvalPlus adapter')
    import pyarrow.parquet as pq
    hf = pq.read_table(hf_path).to_pylist()
    official = list(read_jsonl(official_path))
    hf_ids = {str(r['task_id']) for r in hf}
    if len(hf) != spec['rows'] or len(official) != spec['rows'] or hf_ids != {r['task_id'] for r in official}:
        raise ValueError('Hugging Face and official EvalPlus task IDs/counts differ')
    questions = [dict(benchmark=benchmark, task_id=r['task_id'], prompt=r['prompt'], entry_point=r['entry_point']) for r in official]
    directory.mkdir(parents=True, exist_ok=True)
    save_jsonl(directory / 'questions.jsonl', questions)
    # Keep the official serialized inputs unchanged for EvalPlus's own deserializers.
    with gzip.open(official_path, 'rb') as source, (directory / 'official.jsonl').open('wb') as target:
        shutil.copyfileobj(source, target)
    return questions, ['questions.jsonl', 'official.jsonl']


def build_livecodebench(spec, sources, directory, release, start=None, end=None):
    directory.mkdir(parents=True, exist_ok=True)
    questions, ids = [], set()
    temporary = directory / 'tests.jsonl.gz.tmp'
    with gzip.open(temporary, 'wt', encoding='utf-8') as tests:
        for path in sources:
            for row in read_jsonl(path):
                if not date_selected(row['contest_date'], start, end):
                    continue
                task_id = 'LiveCodeBench/' + row['question_id']
                if task_id in ids:
                    raise ValueError('Duplicate LiveCodeBench task ID')
                ids.add(task_id)
                question = dict(benchmark='livecodebench', task_id=task_id, question_id=row['question_id'],
                                prompt=row['question_content'], starter_code=row['starter_code'],
                                contest_date=row['contest_date'], difficulty=row['difficulty'], platform=row['platform'])
                questions.append(question)
                sample = lcb_sample(row)
                tests.write(json.dumps(dict(task_id=task_id, **sample), ensure_ascii=False) + '\n')
    if not questions:
        temporary.unlink()
        raise ValueError('No LiveCodeBench tasks in the selected release/date range')
    temporary.replace(directory / 'tests.jsonl.gz')
    save_jsonl(directory / 'questions.jsonl', questions)
    return questions, ['questions.jsonl', 'tests.jsonl.gz']


def prepare(benchmark, output, cache, source_dir=None, release='release_v6', start=None, end=None):
    lock = lockfile()
    spec = lock['datasets'][benchmark]
    directory = Path(output) / benchmark
    identity = dict(benchmark=benchmark, protocol=PROTOCOL, source=spec,
                    release=release if benchmark == 'livecodebench' else spec['official_release'],
                    start_date=start if benchmark == 'livecodebench' else None,
                    end_date=end if benchmark == 'livecodebench' else None)
    if (directory / 'manifest.json').exists():
        manifest = read_manifest(output, benchmark)
        if any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError('Prepared benchmark identity changed; choose a new --output-dir')
        print(f'[DATA] Reusing {benchmark}: {manifest["rows"]} questions', flush=True)
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Incomplete/existing benchmark directory; choose another --output-dir: {directory}')
    if benchmark == 'livecodebench':
        count = int(release.rsplit('v', 1)[1])
        sources = [source_file(spec, cache, r['file'], r['sha256'], source_dir) for r in spec['files'][:count]]
        questions, files = build_livecodebench(spec, sources, directory, release, start, end)
    else:
        hf = source_file(spec, cache, spec['file'], spec['file_sha256'], source_dir)
        official = official_release(spec, cache, source_dir)
        questions, files = build_evalplus(benchmark, spec, hf, official, directory)
    manifest = dict(**identity, rows=len(questions), files={name: digest(directory / name) for name in files},
                    test_protocol='official public+private tests' if benchmark == 'livecodebench' else 'official EvalPlus base+plus tests',
                    default_samples=8, metric='pass@8')
    if benchmark == 'livecodebench':
        manifest.update(first_contest_date=min(q['contest_date'] for q in questions),
                        last_contest_date=max(q['contest_date'] for q in questions))
    save_json(directory / 'manifest.json', manifest)
    print(f'[DATA] Prepared {benchmark}: {len(questions)} questions', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['all', *BENCHMARKS], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path('data/code_benchmarks'))
    parser.add_argument('--cache-dir', type=Path, default=Path('data/code_benchmarks_sources'))
    parser.add_argument('--source-dir', type=Path, help='Offline source files with their original names/layout')
    parser.add_argument('--lcb-release', choices=[f'release_v{i}' for i in range(1, 7)], default='release_v6')
    parser.add_argument('--start-date', help='Inclusive LCB contest date, YYYY-MM-DD')
    parser.add_argument('--end-date', help='Inclusive LCB contest date, YYYY-MM-DD')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for name in ('output_dir', 'cache_dir', 'source_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, project_absolute(getattr(args, name)))
    from datetime import date
    if args.start_date:
        date.fromisoformat(args.start_date)
    if args.end_date:
        date.fromisoformat(args.end_date)
    if args.start_date and args.end_date and args.start_date > args.end_date:
        parser.error('--start-date must not be after --end-date')
    selected = BENCHMARKS if args.benchmark == 'all' else [args.benchmark]
    if args.dry_run:
        print(json.dumps(dict(benchmarks=list(selected), metric='pass@8', lcb_release=args.lcb_release,
                              start_date=args.start_date, end_date=args.end_date,
                              locked_sources=lockfile()['datasets']), indent=2))
        return
    for benchmark in selected:
        prepare(benchmark, args.output_dir, args.cache_dir, args.source_dir, args.lcb_release, args.start_date, args.end_date)


if __name__ == '__main__':
    main()

