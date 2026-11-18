"""Evaluate the untrained Qwen3-4B-Base on the five Table 3 math benchmarks."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
from disk_space import require_space, SpaceCheckedWriter
from filelock import FileLock
from model_assets import read_models
from package_results import archive_path, package_results
from project_paths import ROOT, project_absolute
from rq4_results import five_benchmark_scores, read


# Published question counts, in the same benchmark_index namespace as the trained runs.
QUESTIONS = {'math500': 500, 'hmmt25_feb': 30, 'hmmt26_feb': 33, 'aime26': 30, 'minerva': 272}


def paper_questions(source):
    rows = [row for row in evaluate.read_rows(source) if row['benchmark'] in QUESTIONS]
    for name, count in QUESTIONS.items():
        indices = [row['benchmark_index'] for row in rows if row['benchmark'] == name]
        if len(indices) != count or set(indices) != set(range(count)):
            raise ValueError(f'Expected {count} distinct {name} questions indexed 0 through {count - 1}')
    return rows


def base_snapshot(spec, cache):
    from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download, try_to_load_from_cache
    from huggingface_hub.errors import LocalEntryNotFoundError
    cache = project_absolute(cache or constants.HF_HUB_CACHE)
    options = dict(revision=spec['revision'], cache_dir=cache)
    repo = spec['repo']
    try:
        folder = Path(snapshot_download(repo, local_files_only=True, **options))
        index = folder/'model.safetensors.index.json'
        files = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json']
        if index.is_file():
            shards = set(read(index)['weight_map'].values())
            if not shards:
                raise ValueError(f'Empty Base model weight index: {index}')
            files.extend(shards)
        else:
            files.append('model.safetensors')
        if all((folder/name).is_file() for name in files):
            return folder
    except LocalEntryNotFoundError:
        pass
    # Check the actual cache volume, even when --output is on a different mount.
    info = HfApi().model_info(repo, revision=spec['revision'], files_metadata=True)
    if info.sha != spec['revision']:
        raise ValueError('Hub returned a different Base model revision')
    files = [f for f in info.siblings if '/' not in f.rfilename and
             f.rfilename.endswith(('.json', '.safetensors', '.txt', '.jinja', '.model'))]
    missing = [f for f in files if not isinstance(try_to_load_from_cache(repo, f.rfilename, **options), str)]
    if not files or any(f.size is None for f in missing):
        raise ValueError('Cannot determine download size for pinned Base snapshot')
    needed = sum(f.size for f in missing)
    require_space(cache, needed)
    print(f'[DISK] Base model cache {cache}: {needed / 1024**3:.2f} GiB to download', flush=True)
    for f in missing:
        require_space(cache, f.size)
        hf_hub_download(repo, f.rfilename, **options)
    return Path(snapshot_download(repo, local_files_only=True, **options))


def both_scores(row):
    strict = evaluate.score_row(row)
    lenient = evaluate.score_row_lenient(row)
    return dict(benchmark=row['benchmark'], benchmark_index=row['benchmark_index'],
                acc_list=lenient['acc_list'], strict_last_box_acc_list=strict['acc_list'],
                lenient_hit_sources=lenient['lenient_hit_sources'])


def snapshot_hashes(model):
    return {p.name: evaluate.digest(p) for p in sorted(model.iterdir()) if p.is_file()}


def generation_rows(output, inputs, identity, args, cache_dir):
    raw = output/'responses.jsonl.gz'
    meta = output/'responses.meta.json'
    partial = output/'responses.jsonl.gz.tmp'
    if meta.exists():
        record = read(meta)
        if record['identity'] != identity:
            raise ValueError('Base generation identity changed; choose a new --output directory')
        # A stop between publishing the receipt and renaming the verified payload is recoverable.
        path = raw if raw.exists() else partial
        if evaluate.digest(path) != record['sha256']:
            raise ValueError('Base response checksum mismatch')
        if path == partial:
            partial.replace(raw)
        with gzip.open(raw, 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
    else:
        if raw.exists():
            raise ValueError(f'Missing receipt for existing responses: {raw}')
        # Keep a verified previous-format generation file in place; never copy old answers.
        old = output/'mean8.generations.jsonl'
        receipt = output/'mean8.generations.meta.json'
        if old.exists() or receipt.exists():
            saved = read(receipt)
            old_questions = output/'questions.jsonl'
            if evaluate.read_rows(old_questions) != inputs:
                raise ValueError('Saved Base questions differ from the current dataset')
            model = base_snapshot(identity['model'], cache_dir)
            previous = dict(schema=1, dataset_sha256=evaluate.digest(old_questions), sampling=identity['sampling'],
                            prompt_encoding=identity['prompt_encoding'],
                            model_sha256=snapshot_hashes(model))
            rows = evaluate.load_cache(old, receipt, previous)
            return rows, old, saved['sha256']
        if (output/'scores.jsonl').exists() or any(output.glob('mean8*.jsonl')):
            raise ValueError('Missing generation cache for existing Base results; restore the cache or use a new --output')
        model = base_snapshot(identity['model'], cache_dir)
        require_space(output)
        args.model_path = str(model)
        model_hashes = snapshot_hashes(model)
        rows = evaluate.generate(args, inputs)
        try:
            with SpaceCheckedWriter(partial) as writer:
                with gzip.GzipFile(fileobj=writer, mode='wb', compresslevel=3) as stream:
                    for row in rows:
                        stream.write((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
                writer.flush()
                os.fsync(writer.fileno())
            record = dict(identity=identity, model_sha256=model_hashes, sha256=evaluate.digest(partial), rows=len(rows))
            require_space(output, len(json.dumps(record).encode('utf-8')) + 4096)
            evaluate.save_json(meta, record)
        except BaseException:
            if not meta.is_file():
                partial.unlink(missing_ok=True)
            raise
        partial.replace(raw)
    if len(rows) != len(inputs) or any(len(r['responses']) != args.n for r in rows):
        raise ValueError('Incomplete Base response file')
    return rows, raw, record['sha256']


def run(args, output):
    inputs = paper_questions(project_absolute(args.math_benchmarks))
    dataset_hash = hashlib.sha256()
    for row in inputs:
        dataset_hash.update((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8'))
    spec = read_models('4b')['student']
    settings = argparse.Namespace(n=8, temperature=1., top_p=1., seed=42, max_prompt_tokens=1024,
                                  max_tokens=8192, max_num_seqs=args.eval_max_num_seqs)
    identity = dict(model=spec, dataset_sha256=dataset_hash.hexdigest(), sampling=evaluate.sampling(settings),
                    prompt_encoding=evaluate.CHAT_PROMPT_ENCODING)
    sys.path.insert(0, str(ROOT/'runtime'))
    rows, raw, checksum = generation_rows(output, inputs, identity, settings,
                                          project_absolute(args.cache_dir) if args.cache_dir else None)
    scores_file = output/'scores.jsonl'
    results = evaluate.grade(rows, scores_file, checksum, args.score_workers, both_scores, scores_only=True)
    summary = dict(scale='4b', strategy='base', model=spec, rollout_rounds=0, optimizer_updates=0,
                   training_wall_seconds=None, questions=len(rows), samples=settings.n * len(rows),
                   questions_sha256=identity['dataset_sha256'], responses_sha256=checksum,
                   responses_file=raw.name, scores_file=scores_file.name,
                   responses_bytes=raw.stat().st_size, scores_bytes=scores_file.stat().st_size)
    for mode, key in [(evaluate.LENIENT_MODE, 'acc_list'), ('strict_last_box', 'strict_last_box_acc_list')]:
        scores = {}
        for name in QUESTIONS:
            group = [r[key] for r in results if r['benchmark'] == name]
            samples = sum(map(len, group))
            scores[name] = dict(problems=len(group), samples=samples, mean_at_8=sum(map(sum, group))/samples)
        summary[mode] = five_benchmark_scores(scores)
        print(f"4B Base ({mode}), five-benchmark Mean@8: {100 * summary[mode]['five_benchmark_mean_at_8']:.2f}%")
    require_space(output, len(json.dumps(summary).encode('utf-8')) + 4096)
    evaluate.save_json(output/'summary.json', summary)
    print(f"[DISK] {raw.name}: {summary['responses_bytes']:,} bytes; "
          f"{scores_file.name}: {summary['scores_bytes']:,} bytes", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/h200/4b/math-base'))
    parser.add_argument('--math-benchmarks', type=Path, default=Path('data/benchmarks.jsonl'))
    parser.add_argument('--cache-dir', type=Path, help='Hugging Face cache containing the pinned Base snapshot')
    parser.add_argument('--eval-max-num-seqs', type=int, default=64)
    parser.add_argument('--score-workers', type=int, default=evaluate.automatic_workers())
    parser.add_argument('--no-package', action='store_true', help='Skip the standalone results ZIP')
    args = parser.parse_args(argv)
    if min(args.eval_max_num_seqs, args.score_workers) < 1:
        parser.error('Evaluation concurrency and score workers must be positive')

    output = project_absolute(args.output)
    require_space(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output/'.base-evaluation.lock'), timeout=0):
        run(args, output)
        if not args.no_package:
            package_results(output, archive_path(output, '4b'), phase='base-eval')


if __name__ == '__main__':
    # Keep the standalone download's temporary data on the measured Hub cache volume.
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    main()
