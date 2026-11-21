"""Container quotas, native pools and both launcher entry points."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import cpu_resources as cpu
import run_h200_experiment as single
import run_h200_suite as suite


@pytest.fixture
def machine(monkeypatch):
    monkeypatch.setattr(cpu, 'allowed_cpu_ids', lambda: list(range(32, 160)))
    monkeypatch.setattr(cpu, 'cpu_quota', lambda: None)


@pytest.mark.parametrize('gpus,expected', [(1, 4), (2, 4), (4, 8)])
def test_host_128_does_not_create_128_ray_cpus(machine, gpus, expected):
    result = cpu.resolve_cpu_resources(gpus)
    assert result['ray_cpus'] == expected
    assert result['cpu_ids'] == list(range(32, 32 + expected))


def test_quota_and_noncontiguous_affinity(machine, monkeypatch):
    monkeypatch.setattr(cpu, 'allowed_cpu_ids', lambda: [8, 10, 12, 14, 16, 18])
    monkeypatch.setattr(cpu, 'cpu_quota', lambda: 3.5)
    assert cpu.available_cpu_ids() == [8, 10, 12]
    assert cpu.resolve_cpu_resources(1)['ray_cpus'] == 3
    with pytest.raises(ValueError, match='at least 5 Ray CPUs'):
        cpu.resolve_cpu_resources(4)
    with pytest.raises(ValueError, match='exceeds'):
        cpu.resolve_cpu_resources(1, 4)


def test_explicit_selection_applies_in_dry_run_too(machine):
    args = single.parser().parse_args(['--scale', '1.7b', '--dry-run', '--cpu-ids', '40,44'])
    result = single.build_plan(args, {})
    assert result['cpu_ids'] == [40, 44]
    assert result['ray_cpus'] == 2
    assert result['train'][result['train'].index('--parameter-token-samples') + 1] == '64'
    with pytest.raises(ValueError, match='distinct available'):
        cpu.resolve_cpu_resources(1, cpu_ids='0,1,2,3')
    with pytest.raises(ValueError, match='distinct available'):
        cpu.resolve_cpu_resources(1, cpu_ids='40,40')


def test_suite_cap_and_isolation(machine, tmp_path):
    args = suite.parser().parse_args(['--output', str(tmp_path)])
    plan = suite.build_suite(args, {}, list(range(128)), 1024)
    small = [q[0] for q in plan['small_queues']]
    groups = [set(j['cpu_ids']) for j in small] + [set(plan['grade_cpus'])]
    assert sum(map(len, groups)) == len(set.union(*groups))
    assert all(len(j['cpu_ids']) == 4 for j in small)
    assert all(len(j['cpu_ids']) == 8 for j in plan['large'])


def test_ray_rpc_pools_are_bounded_before_child_start(machine):
    plan = single.build_plan(single.parser().parse_args(['--scale', '1.7b']), {})
    env = single.child_environment(plan, {'OMP_NUM_THREADS': '128', 'RAY_num_server_call_thread': '32'})
    for key in ('RAY_num_server_call_thread', 'RAY_gcs_server_rpc_server_thread_num',
                'RAY_gcs_server_rpc_client_thread_num'):
        assert env[key] == '2'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        assert env[key] == '1'
    assert env['TOKENIZERS_PARALLELISM'] == 'false'


def test_v2_nested_parent_quota(tmp_path):
    root = tmp_path / 'cg'
    child = root / 'team/job'
    child.mkdir(parents=True)
    (root / 'cpu.max').write_text('max 100000')
    (child.parent / 'cpu.max').write_text('350000 100000')
    (child / 'cpu.max').write_text('800000 100000')
    membership = tmp_path / 'membership'
    membership.write_text('0::/team/job\n')
    assert cpu.cpu_quota(root, membership) == 3.5


def test_v1_quota_and_unlimited(tmp_path):
    root = tmp_path / 'cg'
    child = root / 'cpu/team'
    child.mkdir(parents=True)
    (child / 'cpu.cfs_quota_us').write_text('400000')
    (child / 'cpu.cfs_period_us').write_text('100000')
    membership = tmp_path / 'membership'
    membership.write_text('2:cpu,cpuacct:/team\n')
    assert cpu.cpu_quota(root, membership) == 4
    (child / 'cpu.cfs_quota_us').write_text('-1')
    assert cpu.cpu_quota(root, membership) is None


def test_grade_can_keep_training_budget_without_starting_ray(machine):
    result = cpu.resolve_cpu_resources(4, 8, '32,33', training=False)
    assert result['cpu_ids'] == [32, 33]
