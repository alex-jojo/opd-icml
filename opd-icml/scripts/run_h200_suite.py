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
import time
import uuid

from run_h200_experiment import ROOT, EXPERIMENTS, select_devices
from wandb_config import load_wandb_environment
from project_paths import project_absolute, project_relative
from package_results import archive_path, package_results
from cleanup_model_artifacts import print_disk_space

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
    p.add_argument('--output', type=Path, default=Path('runs/h200-suite'))
    p.add_argument('--scales', nargs='+', choices=EXPERIMENTS, default=SMALL + LARGE)
    p.add_argument('--max-parallel-small', type=int, choices=[1, 2, 3, 4], default=4)
    p.add_argument('--profile', choices=['throughput', 'conservative'], default='throughput')
    p.add_argument('--parameter-token-gradients', choices=['sampled', 'exact-all'], default='sampled')
    p.add_argument('--parameter-token-samples', type=int, default=256)
    p.add_argument('--gradient-cache-total-gib', type=float, default=64)
    p.add_argument('--ray-object-store-total-gib', type=float, default=16)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--keep-model-files', action='store_true', help='Keep all completed checkpoint/merged weights; default cleans each experiment')
    p.add_argument('--no-package', action='store_true', help='Skip the final combined experiment-data ZIP')
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
    # TaskRunner holds one Ray CPU while waiting for all GPU-worker bundles.
    minimum_training_cpus = max(EXPERIMENTS[s].gpus for s in args.scales) + 1
    grade_count = min(8, max(2, len(cpus) // 8), len(cpus) - minimum_training_cpus)
    if grade_count < 1:
        raise ValueError(f'Suite needs at least {minimum_training_cpus + 1} CPU threads '
                         'for training and concurrent grading')
    grade_cpus, training_cpus = cpus[-grade_count:], cpus[:-grade_count]
    selected_small = [s for s in SMALL if s in args.scales]
    # Conservative admission reserve for model states, diagnostic snapshots and
    # merge buffers. Actual memory remains model-dependent; never silently shrink statistics.
    slots = min(args.max_parallel_small, len(selected_small) or 1,
                max(1, len(training_cpus) // 4), max(1, int((free_gib-cache-objects) // 80)))
    run = project_absolute(args.output)
    common = ['--profile', args.profile, '--parameter-token-gradients', args.parameter_token_gradients,
              '--parameter-token-samples', str(args.parameter_token_samples), '--no-package']
    if args.keep_model_files:
        common.append('--keep-model-files')
    def job(scale, ids, cores, concurrency):
        spec = EXPERIMENTS[scale]
        if len(cores) < spec.gpus + 1:
            raise ValueError(f'{scale} requires at least {spec.gpus + 1} training CPUs')
        output = run / scale
        cmd = [sys.executable, 'scripts/run_h200_experiment.py', '--scale', scale,
               '--devices', ','.join(ids), '--output', project_relative(output), *common,
               '--ray-num-cpus', str(len(cores)), '--cpu-ids', ','.join(map(str, cores)),
               '--ray-object-store-gib', str(objects/concurrency),
               '--gradient-cache-gib', str(cache/concurrency),
               '--code-workers', str(max(1, min(8, len(cores)//2))),
               '--score-workers', str(max(1, min(8, len(cores)//2))),
               '--data', 'data/eurus_code_train.parquet' if spec.task == 'code' else 'data/dapo14k_train.parquet']
        if spec.task == 'code':
            cmd += ['--val-data', 'data/eurus_code_validation.parquet']
        grade = list(cmd)
        grade[grade.index('--cpu-ids')+1] = ','.join(map(str, grade_cpus))
        grade[grade.index('--score-workers')+1] = str(len(grade_cpus))
        return dict(scale=scale, devices=ids, cpu_ids=cores, output=project_relative(output),
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
        prepare.append([sys.executable, 'scripts/prepare_dapo14k.py', '--output', 'data/dapo14k_train.parquet'])
    if any(EXPERIMENTS[s].task == 'code' for s in args.scales):
        prepare += [[sys.executable, 'scripts/prepare_eurus_code.py'],
                    [sys.executable, 'scripts/prepare_code_benchmarks.py']]
    prepare += [[sys.executable, 'scripts/prepare_models.py', '--scale', s] for s in args.scales]
    return dict(output=project_relative(run), path_base='project_root', devices=devices, free_host_gib=free_gib,
                results_zip=None if args.no_package else project_relative(archive_path(run)),
                small_parallelism=slots, grade_cpus=grade_cpus,
                gradient_cache_total_gib=cache, ray_object_store_total_gib=objects,
                prepare=prepare, small_queues=queues, large=large)


def print_execution_schedule(plan):
    """Show actual selected device IDs, including reduced-parallelism queues."""
    rows = []
    for queue in plan['small_queues']:
        for position, job in enumerate(queue, 1):
            rows.append((job['scale'], ','.join(job['devices']),
                         f"stage 1, GPU queue position {position}"))
    for stage, job in enumerate(plan['large'], 2):
        rows.append((job['scale'], ','.join(job['devices']),
                     f"stage {stage}, all selected GPUs"))
    print("\nModel                    | GPU IDs             | Execution order", flush=True)
    print("-" * 88, flush=True)
    for scale, ids, order in rows:
        print(f"{scale:<24} | {ids:<19} | {order}", flush=True)
    print(f"Results: {plan['output']}\n", flush=True)


def resolve_gpu_uuids(devices, environment):
    """Resolve CUDA's selected device order in a short-lived process, never NVML index order."""
    env = environment.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    probe = [sys.executable, '-c',
        'import json, torch; '
        f'assert torch.cuda.device_count() == {len(devices)}, "Selected GPUs must be available"; '
        'print(json.dumps([str(torch.cuda.get_device_properties(i).uuid) '
        'for i in range(torch.cuda.device_count())]))']
    result = subprocess.run(probe, cwd=ROOT, env=env, check=True, capture_output=True,
                            text=True, encoding='utf-8', timeout=30)
    # Torch 2.6 exposes the CUDA UUID as an 8-4-4-4-12 string, without GPU-.
    identities = [str(uuid.UUID(value.removeprefix('GPU-'))) for value in json.loads(result.stdout)]
    if len(identities) != len(devices) or len(set(identities)) != len(devices):
        raise ValueError('Selected GPU aliases must resolve to distinct physical devices')
    return identities


def gpu_lock_name(identity):
    identity = str(uuid.UUID(identity.removeprefix('GPU-')))
    return 'opd-h200-' + hashlib.sha256(identity.encode()).hexdigest()[:16] + '.lock'


def _signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
        return True
    except ProcessLookupError:
        return False


def terminate_groups(processes, grace_seconds):
    """Wait for process GROUPS, even when their original launchers already exited."""
    groups = [process for process in processes if _signal_group(process, signal.SIGTERM)]
    try:
        deadline = time.monotonic() + grace_seconds
        while groups:
            # Reap our direct children so an exited group leader does not keep
            # the group alive by remaining a zombie.
            for process in groups:
                process.poll()
            groups = [process for process in groups if _signal_group(process, 0)]
            remaining = deadline - time.monotonic()
            if not groups or remaining <= 0:
                break
            time.sleep(min(.05, remaining))
    finally:
        # A live group must be killed even if its leader exited immediately,
        # or another interruption arrived while we were waiting for cleanup.
        for process in groups:
            _signal_group(process, signal.SIGKILL)
        for process in processes:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait(timeout=1)


class Processes:
    def __init__(self, termination_grace_seconds=10):
        self.lock = threading.Lock()
        self.active = set()
        self.cancelled = False
        self.termination_grace_seconds = termination_grace_seconds

    def run(self, command, log, cpu_only=False):
        env = load_wandb_environment(os.environ)
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
                    raise subprocess.CalledProcessError(code, command)
            finally:
                try:
                    # Keep the group tracked until its descendants are gone or
                    # SIGKILL has been sent, on success as well as failure.
                    terminate_groups([process], self.termination_grace_seconds)
                finally:
                    with self.lock:
                        self.active.discard(process)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            processes = list(self.active)
        try:
            terminate_groups(processes, self.termination_grace_seconds)
        finally:
            with self.lock:
                self.active.difference_update(processes)


def schedule(plan, runner):
    """GPU completion releases its slot; one bounded CPU grading queue overlaps it."""
    root = project_absolute(plan['output'])
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
        if args.dry_run:
            print(json.dumps(plan, indent=2), flush=True)
            return
        print_execution_schedule(plan)
        print_disk_space(plan['output'])
        if os.name != 'posix':
            raise ValueError('Execution requires a Linux H200 server')
        import fcntl
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Suite interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        # CUDA indices, UUIDs and UUID prefixes share the same physical-device lock.
        plan['device_uuids'] = resolve_gpu_uuids(plan['devices'], os.environ)
        for identity in sorted(plan['device_uuids']):
            stream = locks.enter_context((Path(tempfile.gettempdir())/gpu_lock_name(identity)).open('a'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f'GPU {identity} is already owned by another OPD suite')
        root = project_absolute(plan['output'])
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
        # Compression is CPU-only. Release GPU ownership before creating the ZIP.
        locks.close()
        if plan['results_zip']:
            package_results(root, plan['results_zip'], phase='suite-complete')
    except BaseException:
        runner.cancel()
        raise
    finally:
        locks.close()


if __name__ == '__main__':
    main()
