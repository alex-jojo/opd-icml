"""One independent vLLM process per visible GPU, with resumable disjoint tasks.

Keeps any serial answer prefix unchanged. Each shard journals its own answers;
only a validated complete merge becomes the ordinary generations.jsonl file.
"""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import time
from tqdm import tqdm
from code_benchmarks import digest, read_jsonl, save_json, save_jsonl, validate_generations
from watch_code_progress import LineCounter


def visible_devices():
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = [part.strip() for part in raw.split(',')]
    if not raw or any(not part or part == '-1' for part in devices) or len(set(devices)) != len(devices):
        raise ValueError('--parallel-mode data requires an explicit unique CUDA_VISIBLE_DEVICES list, e.g. 0,1,2,3')
    return devices


def population_hash(questions):
    payload = json.dumps(questions, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def shard_directory(directory, count, index):
    return Path(directory) / 'generation_shards' / f'workers_{count}' / f'shard_{index:03d}'


def prepare_plan(directory, questions, samples, count):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    legacy = directory / 'generation.partial.jsonl'
    seed_path = directory / 'generation.seed.jsonl'
    seed = list(read_jsonl(seed_path)) if seed_path.exists() else (list(read_jsonl(legacy)) if legacy.exists() else [])
    validate_generations(questions[:len(seed)], seed, samples)
    if legacy.exists() and list(read_jsonl(legacy)) != seed:
        raise ValueError('Serial answers changed while data-parallel shards exist; do not run two jobs into one directory')
    if not seed_path.exists():
        save_jsonl(seed_path, seed)
    plan = dict(version=1, workers=count, samples=samples, questions_sha256=population_hash(questions),
                seed_sha256=digest(seed_path), reused_questions=len(seed), total_questions=len(questions))
    path = directory / 'generation.plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8')) != plan:
        raise ValueError('Data-parallel plan changed; resume with the same number of GPUs or choose a new output directory')
    save_json(path, plan)
    return plan, seed


def shard_questions(directory, questions, samples, count, index):
    if not 0 <= index < count:
        raise ValueError('Invalid generation shard index')
    directory = Path(directory)
    plan = json.loads((directory / 'generation.plan.json').read_text(encoding='utf8'))
    seed_path = directory / 'generation.seed.jsonl'
    if (plan['workers'] != count or plan['samples'] != samples or
            plan['questions_sha256'] != population_hash(questions) or plan['seed_sha256'] != digest(seed_path)):
        raise ValueError('Generation shard plan/input identity mismatch')
    seed = list(read_jsonl(seed_path))
    validate_generations(questions[:len(seed)], seed, samples)
    return [q for i, q in enumerate(questions) if i >= len(seed) and i % count == index]


def merge_shards(directory, questions, samples, count):
    directory = Path(directory)
    seed = list(read_jsonl(directory / 'generation.seed.jsonl'))
    legacy = directory / 'generation.partial.jsonl'
    if legacy.exists() and list(read_jsonl(legacy)) != seed:
        raise ValueError('Serial answer prefix changed during parallel generation')
    combined = {row['task_id']: row for row in seed}
    for index in range(count):
        subset = shard_questions(directory, questions, samples, count, index)
        path = shard_directory(directory, count, index) / 'generations.jsonl'
        rows = list(read_jsonl(path)) if path.exists() else []
        validate_generations(subset, rows, samples)
        for row in rows:
            if row['task_id'] in combined:
                raise ValueError('Duplicate question across generation shards')
            combined[row['task_id']] = row
    if set(combined) != {q['task_id'] for q in questions}:
        raise ValueError('Incomplete generation shard coverage')
    ordered = [combined[q['task_id']] for q in questions]
    validate_generations(questions, ordered, samples)
    save_jsonl(directory / 'generations.jsonl', ordered)
    return ordered


def generate_parallel(command, directory, questions, samples, devices):
    directory = Path(directory)
    output = directory / 'generations.jsonl'
    if output.exists():
        validate_generations(questions, list(read_jsonl(output)), samples)
        return
    count = len(devices)
    if count < 1:
        raise ValueError('At least one generation GPU is required')
    plan, seed = prepare_plan(directory, questions, samples, count)
    processes = []
    counters = [LineCounter() for _ in devices]
    started = time.monotonic()
    execution = dict(mode='data', devices=devices, tensor_parallel_per_worker=1,
        workers=count, reused_questions=len(seed), question_partition='global index modulo workers, excluding saved prefix',
        note='Sampling protocol is unchanged; stochastic responses are not guaranteed bitwise identical across batch/parallel layouts',
        status='running', logs={})
    save_json(directory / 'generation.execution.json', execution)
    def saved_count():
        value = len(seed)
        for index, counter in enumerate(counters):
            folder = shard_directory(directory, count, index)
            path = folder / 'generations.jsonl'
            value += counter.read(path if path.exists() else folder / 'generation.partial.jsonl')
        return value
    try:
        with ExitStack() as stack:
            progress = stack.enter_context(tqdm(total=len(questions), initial=len(seed),
                desc=f'{directory.name} | {count} GPUs', unit='question', dynamic_ncols=True))
            for index, device in enumerate(devices):
                subset = shard_questions(directory, questions, samples, count, index)
                folder = shard_directory(directory, count, index)
                folder.mkdir(parents=True, exist_ok=True)
                if not subset:
                    save_jsonl(folder / 'generations.jsonl', [])
                    continue
                logfile = folder / 'worker.log'
                execution['logs'][str(index)] = str(logfile)
                stream = stack.enter_context(logfile.open('a', encoding='utf8'))
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = device
                env.setdefault('OMP_NUM_THREADS', '1')
                process = subprocess.Popen(command+['--shard-count', str(count), '--shard-index', str(index)],
                                           env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == 'posix')
                processes.append((process, logfile))
            while True:
                progress.update(saved_count()-progress.n)
                failures = [(p.returncode, log) for p, log in processes if p.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f'Generation worker failed; completed answers are resumable. Logs: {failures}')
                if all(p.poll() == 0 for p, _ in processes):
                    break
                time.sleep(.5)
            merge_shards(directory, questions, samples, count)
            progress.update(len(questions)-progress.n)
        execution['status'] = 'complete'
    finally:
        # Failure/interrupt also stops vLLM grandchildren in our OWN process groups.
        if execution['status'] != 'complete' and os.name == 'posix':
            for process, _ in processes:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if execution['status'] != 'complete':
            execution['status'] = 'interrupted_or_failed'
        execution['seconds'] = time.monotonic()-started
        save_json(directory / 'generation.execution.json', execution)
