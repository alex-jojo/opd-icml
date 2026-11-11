"""Resource ownership, failure propagation and unchanged scientific budgets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_h200_experiment as h200
import run_h200_suite as suite


def plan_for(tmp_path, extra=(), free=512, cpus=64):
    args = suite.parser().parse_args(['--output', str(tmp_path), *extra])
    return suite.build_suite(args, {'CUDA_VISIBLE_DEVICES': '3,1,2,0'}, list(range(cpus)), free)


def value(command, option):
    return command[command.index(option)+1]


def test_parallel_plan_has_disjoint_gpu_cpu_and_aggregate_memory_budgets(tmp_path):
    plan = plan_for(tmp_path)
    assert plan['small_parallelism'] == 4
    small = [q[0] for q in plan['small_queues']]
    assert [j['devices'] for j in small] == [['3'], ['1'], ['2'], ['0']]
    groups = [set(j['cpu_ids']) for j in small]+[set(plan['grade_cpus'])]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 64
    for option, cap in [('--gradient-cache-gib', 64), ('--ray-object-store-gib', 16)]:
        assert sum(float(value(j['command'], option)) for j in small) <= cap
    for job in small+plan['large']:
        assert value(job['command'], '--parameter-token-samples') == '256'
        assert value(job['command'], '--stage') == 'train-generate'
    assert all(j['devices'] == ['3', '1', '2', '0'] for j in plan['large'])


def test_low_memory_reduces_parallelism_without_reducing_diagnostics(tmp_path):
    plan = plan_for(tmp_path, free=180, cpus=24)
    assert plan['small_parallelism'] == 1
    assert len(plan['small_queues'][0]) == 4
    assert all(value(j['command'], '--parameter-token-samples') == '256' for j in plan['small_queues'][0])


@pytest.mark.parametrize('scale', list(h200.EXPERIMENTS))
def test_profiles_resolve_through_real_launcher_without_changing_budget(scale, tmp_path):
    plans = []
    for profile in ('conservative', 'throughput'):
        args = h200.parser().parse_args(['--scale', scale, '--profile', profile, '--output', str(tmp_path/scale)])
        plan = h200.build_plan(args, {})
        env = h200.child_environment(plan, {'RAY_ADDRESS': 'unrelated-cluster'})
        assert env['RAY_ADDRESS'] == 'local'
        result = subprocess.run(plan['train']+['--dry-run'], cwd=ROOT, env={**os.environ, **env},
                                capture_output=True, text=True, encoding='utf-8')
        assert result.returncode == 0, result.stderr
        actual = json.loads(result.stdout)
        assert actual['parameter_token_gradients']['sample_size'] == 256
        assert actual['rounds_cap'] == 106 and actual['rollout_batch'] == actual['mini_batch_size'] == 128
        assert actual['updates_per_round'] == 1 and actual['learning_rate'] == 1e-5
        assert 'ray_kwargs.ray_init.address' in actual['command']
        assert 'local' in actual['command']
        plans.append(actual)
    micro, teacher, sequences, offload = h200.PRESETS[scale]
    assert plans[1]['micro_batch_size_per_gpu'] == micro
    assert plans[1]['teacher_micro_batch_size_per_gpu'] == teacher
    assert plans[1]['rollout_max_num_seqs'] == sequences
    assert plans[1]['optimizer_offload'] == (offload == 'on')
    assert plans[1]['teacher_offload'] == (offload == 'on')
    assert plans[0]['micro_batch_size_per_gpu'] == plans[0]['teacher_micro_batch_size_per_gpu'] == 1
    assert plans[0]['optimizer_offload'] and plans[0]['teacher_offload']


def test_new_residency_and_ray_settings_compose_with_hydra(tmp_path):
    from hydra import compose, initialize_config_dir
    import shlex
    args = h200.parser().parse_args(['--scale', '8b', '--output', str(tmp_path/'run')])
    plan = h200.build_plan(args, {})
    result = subprocess.run(plan['train']+['--dry-run'], cwd=ROOT,
                            capture_output=True, text=True, encoding='utf-8', check=True)
    command = shlex.split(json.loads(result.stdout)['command'])
    overrides = [v for v in command if v.startswith('++')]
    search = 'hydra.searchpath=' + json.dumps(['file://'+(ROOT/'runtime/verl/trainer/config').as_posix()], ensure_ascii=False)
    with initialize_config_dir(config_dir=str(ROOT/'configs/8b'), version_base=None):
        config = compose(config_name='paired-full', overrides=[search, *overrides])
    assert config.actor_rollout_ref.ref.opd_stage_offload
    assert config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == 2
    assert config.ray_kwargs.ray_init.address == 'local'
    assert config.ray_kwargs.ray_init.object_store_memory == 4*1024**3


def test_grade_stage_never_probes_gpu_merges_or_trains(tmp_path, monkeypatch):
    args = h200.parser().parse_args(['--scale', 'deepseek-coder-33b', '--stage', 'grade', '--output', str(tmp_path)])
    plan = h200.build_plan(args, {})
    calls = []
    monkeypatch.setattr(h200, 'run_command', lambda cmd, env, log: calls.append((cmd, env)))
    h200.execute(args, plan, h200.child_environment(plan, {}))
    assert len(calls) == 1
    assert calls[0][0] == plan['evaluate']+['--grade-only']
    assert calls[0][1]['CUDA_VISIBLE_DEVICES'] == ''


def test_scheduler_releases_gpu_while_grade_runs_and_waits_before_large(tmp_path):
    plan = plan_for(tmp_path)
    lock = threading.Lock()
    active = set()
    completed_small = set()
    grading = threading.Event()
    large_started = threading.Event()
    jobs = {j['scale']: j for q in plan['small_queues'] for j in q}
    jobs.update({j['scale']: j for j in plan['large']})
    class Fake:
        def run(self, command, log, cpu_only=False):
            scale = value(command, '--scale')
            if cpu_only:
                grading.set()
                assert large_started.wait(10), 'grading blocked GPU stage scheduling'
                return
            devices = set(jobs[scale]['devices'])
            with lock:
                assert not active & devices
                active.update(devices)
                if scale in suite.LARGE:
                    assert completed_small == set(suite.SMALL)
                    assert grading.wait(10)
                    large_started.set()
            with lock:
                active.difference_update(devices)
                if scale in suite.SMALL:
                    completed_small.add(scale)
        def cancel(self):
            pytest.fail('Successful suite must not cancel')
    suite.schedule(plan, Fake())
    assert set(json.loads((tmp_path/'status.json').read_text()).values()) == {'complete'}


def test_failure_cancels_and_does_not_launch_large(tmp_path):
    plan = plan_for(tmp_path, ['--max-parallel-small', '1'])
    calls = []
    class Fake:
        cancelled = False
        def run(self, command, log, cpu_only=False):
            calls.append(value(command, '--scale'))
            raise subprocess.CalledProcessError(7, command)
        def cancel(self):
            self.cancelled = True
    runner = Fake()
    with pytest.raises(subprocess.CalledProcessError):
        suite.schedule(plan, runner)
    assert runner.cancelled and calls == ['deepseek-coder-1.3b']


def test_suite_dry_run_has_no_side_effects(tmp_path, monkeypatch, capsys):
    dest = tmp_path/'not-created'
    monkeypatch.setattr(sys, 'argv', ['run_h200_suite.py', '--dry-run', '--output', str(dest)])
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(suite, 'available_resources', lambda: (list(range(64)), 512))
    suite.main()
    assert json.loads(capsys.readouterr().out)['devices'] == ['0', '1', '2', '3']
    assert not dest.exists()
