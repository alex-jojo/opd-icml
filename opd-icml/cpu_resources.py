"""Bound launcher CPU resources using affinity and visible cgroup CPU quotas."""
import math
import os
from pathlib import Path


def allowed_cpu_ids():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))


def cgroup_directories(controller, root=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    """Include namespace root and visible ancestors of this process's group."""
    roots = [root] if controller == 'v2' else [root / controller, root / 'cpu,cpuacct']
    paths = set(roots)
    try:
        lines = membership.read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        if not ((controller == 'v2' and not controllers) or controller in controllers.split(',')):
            continue
        # Host-relative paths may be hidden by a cgroup namespace; root still applies.
        if '..' in Path(relative).parts:
            continue
        for base in roots:
            current = base / relative.lstrip('/')
            while current != base:
                paths.add(current)
                current = current.parent
    return sorted(paths)


def cpu_quota(root=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    limits = []
    for directory in cgroup_directories('v2', root, membership):
        try:
            quota, period = (directory / 'cpu.max').read_text().split()
            if quota != 'max' and int(quota) > 0 and int(period) > 0:
                limits.append(int(quota) / int(period))
        except (OSError, ValueError):
            pass
    for directory in cgroup_directories('cpu', root, membership):
        try:
            quota = int((directory / 'cpu.cfs_quota_us').read_text())
            period = int((directory / 'cpu.cfs_period_us').read_text())
            if quota > 0 and period > 0:
                limits.append(quota / period)
        except (OSError, ValueError):
            pass
    return min(limits) if limits else None


def default_ray_cpus(gpus):
    return max(4, 2 * gpus)


def bounded_thread_environment(environment):
    """Set before starting Ray/native libraries; logical CPU resources are insufficient.

    Ray 2.46/2.48 use hardware_concurrency()/4 for these RPC pools, independent of
    ray.init(num_cpus). Small pools fit this project's GPU-bound local actors.
    """
    env = environment.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    for key in ('RAY_num_server_call_thread', 'RAY_gcs_server_rpc_server_thread_num',
                'RAY_gcs_server_rpc_client_thread_num'):
        env[key] = '2'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    return env


def available_cpu_ids():
    ids = allowed_cpu_ids()
    quota = cpu_quota()
    return ids if quota is None else ids[:max(1, math.floor(quota))]


def resolve_cpu_resources(gpus, requested=None, cpu_ids=None, training=True):
    allowed = allowed_cpu_ids()
    selected = allowed
    if cpu_ids is not None:
        try:
            selected = [int(item.strip()) for item in cpu_ids.split(',')]
        except ValueError:
            raise ValueError('--cpu-ids must be comma-separated available CPU IDs') from None
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            raise ValueError('--cpu-ids must select distinct available CPUs')
    quota = cpu_quota()
    capacity = min(len(selected), max(1, math.floor(quota))) if quota is not None else len(selected)
    if requested is not None and requested < 1:
        raise ValueError('--ray-num-cpus must be positive')
    count = requested if requested is not None else min(default_ray_cpus(gpus), capacity)
    if training and count < gpus + 1:
        raise ValueError(f'Training on {gpus} GPUs requires at least {gpus + 1} Ray CPUs '
                         '(one per GPU worker plus one TaskRunner); increase the container CPU allocation')
    if training and count > capacity:
        raise ValueError(f'--ray-num-cpus={count} exceeds available affinity/quota capacity {capacity}')
    # Grade-only jobs do not start Ray; their suite command can carry a training budget.
    affinity = selected if cpu_ids is not None else selected[:min(count, capacity)]
    return dict(ray_cpus=count, cpu_ids=affinity, visible_cpu_count=len(allowed), cpu_quota=quota)
