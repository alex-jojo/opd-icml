"""Schedule seven H200 paired-FULL studies with bounded CPU, cache and GPU ownership."""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import tempfile
import hashlib

from run_h200_experiment import ROOT, EXPERIMENTS, select_devices

SMALL = ['deepseek-coder-1.3b', '1.7b', '4b', 'deepseek-coder-6.7b']
LARGE = ['8b', 'gemma3-12b', 'deepseek-coder-33b']
GIB = 1024**3


def available_resources():
    import psutil
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    available = psutil.virtual_memory().available
    for limit, used in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            value = Path(limit).read_text().strip()
            if value != 'max':
                available = min(available, max(0, int(value) - int(Path(used).read_text())))
        except (OSError, ValueError):
            pass
    return cpus, available / GIB


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices', help='Four GPU IDs/UUIDs, in scheduling order')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/h200-suite')
    p.add_argument('--scales', nargs='+', choices=EXPERIMENTS, default=SMALL + LARGE)
    p.add_argument('--max-parallel-small', type=int, choices=[1, 2, 3, 4], default=4)
    p.add_argument('--profile', choices=['throughput', 'conservative'], default='throughput')
    p.add_argument('--parameter-token-gradients', choices=['sampled', 'exact-all'], default='sampled')
    p.add_argument('--parameter-token-samples', type=int, default=256)
    p.add_argument('--gradient-cache-total-gib', type=float, default=64)
    p.add_argument('--ray-object-store-total-gib', type=float, default=16)
    p.add_argument('--dry-run', action='store_true')
    return p


def build_suite(args, environment, cpus, free_gib):
    devices = select_devices(4, args.devices, environment).split(',')
    if len(cpus) < 6:
        raise ValueError('Suite requires at least six available CPU threads')
    if len(set(args.scales)) != len(args.scales):
        raise ValueError('Duplicate scales are not allowed')
    if args.parameter_token_samples < 2:
        raise ValueError('Parameter sample count must be at least two')
    cache, objects = args.gradient_cache_total_gib, args.ray_object_store_total_gib
    if not 0 <= cache < float('inf') or not 0 < objects < float('inf'):
        raise ValueError('Cache/Ray budgets must be finite and nonnegative/positive')
    # These are aggregate buffer caps, not a bound on model/checkpoint allocations.
    cache = min(cache, free_gib / 8)
    objects = min(objects, free_gib / 16)
    if objects < 1:
        raise ValueError('Insufficient available host memory')
    grade_count = min(8, max(2, len(cpus) // 8))
    grade_cpus, training_cpus = cpus[-grade_count:], cpus[:-grade_count]
    selected_small = [s for s in SMALL if s in args.scales]
    # Conservative admission reserve for model states, diagnostic snapshots and
    # merge buffers. Actual memory remains model-dependent; never silently shrink statistics.
    slots = min(args.max_parallel_small, len(selected_small) or 1,
                max(1, len(training_cpus) // 4), max(1, int((free_gib-cache-objects) // 80)))
    run = args.output.resolve()
    common = ['--profile', args.profile, '--parameter-token-gradients', args.parameter_token_gradients,
              '--parameter-token-samples', str(args.parameter_token_samples)]
    def job(scale, ids, cores, concurrency):
        spec = EXPERIMENTS[scale]
        output = run / scale
        cmd = [sys.executable, str(ROOT / 'scripts/run_h200_experiment.py'), '--scale', scale,
               '--devices', ','.join(ids), '--output', str(output), *common,
               '--ray-num-cpus', str(len(cores)), '--cpu-ids', ','.join(map(str, cores)),
               '--ray-object-store-gib', str(objects/concurrency),
               '--gradient-cache-gib', str(cache/concurrency),
               '--code-workers', str(max(1, min(8, len(cores)//2))),
               '--score-workers', str(max(1, min(8, len(cores)//2))),
               '--data', str(ROOT / ('data/eurus_code_train.parquet' if spec.task == 'code' else 'data/dapo14k_train.parquet'))]
        if spec.task == 'code':
            cmd += ['--val-data', str(ROOT / 'data/eurus_code_validation.parquet')]
        grade = list(cmd)
        grade[grade.index('--cpu-ids')+1] = ','.join(map(str, grade_cpus))
        grade[grade.index('--score-workers')+1] = str(len(grade_cpus))
        return dict(scale=scale, devices=ids, cpu_ids=cores, output=str(output),
                    command=cmd+['--stage', 'train-generate'],
                    grade=grade+['--stage', 'grade'] if spec.task == 'code' else None)
    # Fixed slot queues keep each GPU/CPU partition owned by one task at a time.
    queues = [[] for _ in range(slots)]
    for i, scale in enumerate(selected_small):
        slot = i % slots
        cores = training_cpus[slot::slots]
        queues[slot].append(job(scale, [devices[slot]], cores, slots))
    large = [job(s, devices, training_cpus, 1) for s in LARGE if s in args.scales]
    prepare = []
    if any(EXPERIMENTS[s].task == 'math' for s in args.scales):
        prepare.append([sys.executable, str(ROOT/'scripts/prepare_dapo14k.py'), '--output', str(ROOT/'data/dapo14k_train.parquet')])
    if any(EXPERIMENTS[s].task == 'code' for s in args.scales):
        prepare += [[sys.executable, str(ROOT/'scripts/prepare_eurus_code.py')],
                    [sys.executable, str(ROOT/'scripts/prepare_code_benchmarks.py')]]
    prepare += [[sys.executable, str(ROOT/'scripts/prepare_models.py'), '--scale', s] for s in args.scales]
    return dict(output=str(run), devices=devices, free_host_gib=free_gib,
                small_parallelism=slots, grade_cpus=grade_cpus,
                gradient_cache_total_gib=cache, ray_object_store_total_gib=objects,
                prepare=prepare, small_queues=queues, large=large)


class Processes:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = set()
        self.cancelled = False

    def run(self, command, log, cpu_only=False):
        env = os.environ.copy()
        if cpu_only:
            env['CUDA_VISIBLE_DEVICES'] = ''
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as out:
            with self.lock:
                if self.cancelled:
                    raise RuntimeError('Suite cancelled')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out,
                    stderr=subprocess.STDOUT, start_new_session=True)
                self.active.add(process)
            try:
                code = process.wait()
                if code:
                    # Stop only this job's descendants, never a global ray stop.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    raise subprocess.CalledProcessError(code, command)
            finally:
                with self.lock:
                    self.active.discard(process)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            processes = list(self.active)
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def schedule(plan, runner):
    """GPU completion releases its slot; one bounded CPU grading queue overlaps it."""
    root = Path(plan['output'])
    state = {}
    def record(scale, status):
        state[scale] = status
        tmp = root/'status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        tmp.replace(root/'status.json')
        print(f'[{status}] {scale}', flush=True)
    gpu = ThreadPoolExecutor(max_workers=plan['small_parallelism'])
    grader = ThreadPoolExecutor(max_workers=1)
    grades = {}
    def submit(pool, job, grading=False):
        record(job['scale'], 'grading' if grading else 'training-and-generating')
        return pool.submit(runner.run, job['grade'] if grading else job['command'],
                           root/'logs'/f'{job["scale"]}-{"grade" if grading else "gpu"}.log', grading)
    def collect_grades():
        for future, job in list(grades.items()):
            if future.done():
                future.result()
                record(job['scale'], 'complete')
                del grades[future]
    def gpu_done(job):
        if job['grade']:
            grades[submit(grader, job, True)] = job
        else:
            record(job['scale'], 'complete')
    try:
        pending = {}
        queues = [list(q) for q in plan['small_queues']]
        for queue in queues:
            if queue:
                job = queue.pop(0)
                pending[submit(gpu, job)] = (job, queue)
        while pending:
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            collect_grades()
            for future in done:
                job, queue = pending.pop(future)
                future.result()
                gpu_done(job)
                if queue:
                    nxt = queue.pop(0)
                    pending[submit(gpu, nxt)] = (nxt, queue)
        for job in plan['large']:
            collect_grades()
            future = submit(gpu, job)
            while not future.done():
                wait([future], timeout=1)
                collect_grades()
            future.result()
            gpu_done(job)
        while grades:
            wait(grades, timeout=1, return_when=FIRST_COMPLETED)
            collect_grades()
    except BaseException:
        record('__suite__', 'failed-or-interrupted; see logs')
        runner.cancel()
        raise
    finally:
        gpu.shutdown(wait=True, cancel_futures=True)
        grader.shutdown(wait=True, cancel_futures=True)


def main():
    p = parser()
    args = p.parse_args()
    runner = Processes()
    locks = ExitStack()
    try:
        cpus, free = available_resources()
        plan = build_suite(args, os.environ, cpus, free)
        print(json.dumps(plan, indent=2), flush=True)
        if args.dry_run:
            return
        if os.name != 'posix':
            raise ValueError('Execution requires a Linux H200 server')
        import fcntl
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Suite interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        # Prevent two copies of this scheduler from claiming the same GPU IDs.
        for device in sorted(plan['devices']):
            name = 'opd-h200-' + hashlib.sha256(device.encode()).hexdigest()[:16] + '.lock'
            stream = locks.enter_context((Path(tempfile.gettempdir())/name).open('a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f'GPU {device} is already owned by another OPD suite')
        root = Path(plan['output'])
        if root.exists() and any(root.iterdir()):
            raise ValueError('Choose an empty suite --output; existing experiments will not be overwritten')
        root.mkdir(parents=True, exist_ok=True)
        runner.run([sys.executable, '-c',
            'import os; os.environ["CUDA_VISIBLE_DEVICES"]=' + repr(','.join(plan['devices'])) + '; '
            'import torch; assert torch.cuda.device_count()==4, "Four selected GPUs must be available"; '
            'print([torch.cuda.get_device_name(i) for i in range(4)])'], root/'logs/devices.log')
        # Preflight all launch plans before expensive preparation.
        for job in [j for q in plan['small_queues'] for j in q] + plan['large']:
            runner.run(job['command']+['--dry-run'], root/'logs'/f'{job["scale"]}-plan.log')
        for i, command in enumerate(plan['prepare']):
            runner.run(command, root/'logs'/f'prepare-{i}.log', cpu_only=True)
        (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
        schedule(plan, runner)
    except BaseException:
        runner.cancel()
        raise
    finally:
        locks.close()


if __name__ == '__main__':
    main()
