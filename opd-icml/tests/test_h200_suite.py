"""Resource ownership, failure propagation and unchanged scientific budgets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

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
    assert plan['small_parallelism'] == 2
    small = [q[0] for q in plan['small_queues']]
    assert [j['devices'] for j in small] == [['3'], ['1']]
    groups = [set(j['cpu_ids']) for j in small]+[set(plan['grade_cpus'])]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 16
    for option, cap in [('--gradient-cache-gib', 64), ('--ray-object-store-gib', 16)]:
        assert sum(float(value(j['command'], option)) for j in small) <= cap
    for job in small+plan['large']:
        assert value(job['command'], '--parameter-token-samples') == '64'
        assert value(job['command'], '--stage') == 'train-generate'
    assert all(j['devices'] == ['3', '1'] for j in plan['large'])


def test_low_memory_reduces_parallelism_without_reducing_diagnostics(tmp_path):
    plan = plan_for(tmp_path, free=180, cpus=24)
    assert plan['small_parallelism'] == 1
    assert len(plan['small_queues'][0]) == 4
    assert all(value(j['command'], '--parameter-token-samples') == '64' for j in plan['small_queues'][0])


@pytest.mark.parametrize('scales', [
    suite.SMALL + suite.LARGE, ['deepseek-coder-6.7b', 'deepseek-coder-33b'],
    ['deepseek-coder-1.3b', '8b', 'deepseek-coder-33b'],
])
def test_suite_schedules_reuse_for_math_and_smallest_coder(tmp_path, scales):
    plan = plan_for(tmp_path, ['--scales', *scales])
    jobs = [job for queue in [*plan['small_queues'], plan['large']] for job in queue]
    reuse = {'1.7b', '4b', '8b', 'gemma3-12b', 'deepseek-coder-1.3b'} & set(scales)
    assert set(plan['reuse_scales']) == reuse
    for scale in scales:
        matching = [job for job in jobs if job['scale'] == scale]
        assert [job['rq4_strategy'] for job in matching] == (['full', 'reuse'] if scale in reuse else ['full'])
        assert matching[0]['evaluate_base'] and matching[0]['shared_early_stop']


@pytest.mark.parametrize('cpus', [6, 7, 8, 16, 64])
def test_cpu_admission_leaves_room_for_worker_bundles_and_task_runner(tmp_path, cpus):
    plan = plan_for(tmp_path, cpus=cpus)
    assert plan['grade_cpus']
    for job in [j for queue in plan['small_queues'] for j in queue] + plan['large']:
        count = int(value(job['command'], '--ray-num-cpus'))
        assert count == len(job['cpu_ids'])
        assert count >= h200.EXPERIMENTS[job['scale']].gpus + 1
        assert not set(job['cpu_ids']) & set(plan['grade_cpus'])
    if cpus == 6:
        assert len(plan['grade_cpus']) == 2
        assert all(len(job['cpu_ids']) == 4 for job in plan['large'])


def test_training_rejects_a_ray_cpu_budget_that_would_deadlock():
    args = h200.parser().parse_args(['--scale', '8b', '--ray-num-cpus', '4'])
    with pytest.raises(ValueError, match='at least 5 Ray CPUs'):
        h200.build_plan(args, {})


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
        assert actual['parameter_token_gradients']['sample_size'] == 64
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


@pytest.mark.parametrize('mask', ['', '-1'])
@pytest.mark.parametrize('scale', ['deepseek-coder-1.3b', 'deepseek-coder-33b'])
def test_cpu_grade_accepts_hidden_gpus_and_skips_the_training_launcher(tmp_path, monkeypatch, capsys, mask, scale):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', mask)
    monkeypatch.setattr(sys, 'argv', ['run_h200_experiment.py', '--scale', scale,
                                    '--stage', 'grade', '--output', str(tmp_path)])
    monkeypatch.setattr(h200, 'select_devices', lambda *args: pytest.fail('CPU grading must not validate GPUs'))
    monkeypatch.setattr(h200.subprocess, 'run', lambda *args, **kwargs: pytest.fail('CPU grading must not resolve training'))
    calls = []
    monkeypatch.setattr(h200, 'run_command', lambda command, env, log: calls.append((command, env)))
    monkeypatch.setattr(h200, 'package_results', lambda *args, **kwargs: None)
    h200.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan['stage_gpu_count'] == plan['evaluation_gpus'] == 0
    assert plan['evaluation_tp'] is None and plan['cuda_visible_devices'] == ''
    assert len(calls) == 1 and '--grade-only' in calls[0][0]
    assert calls[0][1]['CUDA_VISIBLE_DEVICES'] == ''


def test_gpu_aliases_resolve_in_cuda_order_and_share_lock_names(monkeypatch):
    identities = [f'{0xa0 + index:08x}-0000-0000-0000-000000000000' for index in range(4)]
    # CUDA index 0 deliberately differs from the first physical UUID. This
    # catches implementations that confuse CUDA and nvidia-smi index order.
    selected = [identities[3], identities[1], identities[2], identities[0]]
    aliases = [['0', '1', '2', '3'], ['GPU-' + value for value in selected],
               ['GPU-' + value[:32] for value in selected]]
    calls = []

    def fake_probe(command, **kwargs):
        calls.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
        assert 'get_device_properties(i).uuid' in command[-1]
        return SimpleNamespace(stdout=json.dumps(selected))

    monkeypatch.setattr(suite.subprocess, 'run', fake_probe)
    results = [suite.resolve_gpu_uuids(values, {'CUDA_VISIBLE_DEVICES': 'unrelated'}) for values in aliases]
    assert results == [selected] * len(aliases)
    assert calls == [','.join(values) for values in aliases]
    assert [suite.gpu_lock_name(value) for value in selected] == [
        suite.gpu_lock_name('GPU-' + value.upper()) for value in selected]


def test_resolved_duplicate_physical_devices_are_rejected(monkeypatch):
    identity = '00000000-0000-0000-0000-000000000001'
    monkeypatch.setattr(suite.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps([identity] * 4)))
    with pytest.raises(ValueError, match='distinct physical devices'):
        suite.resolve_gpu_uuids(['0', '1', '2', '3'], {})


@pytest.mark.parametrize('cancel', [False, True])
def test_cleanup_kills_a_surviving_group_even_when_its_leader_has_exited(tmp_path, monkeypatch, cancel):
    process = SimpleNamespace(pid=4321, poll=lambda: 7, wait=lambda **kwargs: 7)
    signals = []
    alive = True
    monkeypatch.setattr(suite.signal, 'SIGKILL', 9, raising=False)
    kill_signal = suite.signal.SIGKILL

    def signal_group(pid, sig):
        nonlocal alive
        assert pid == process.pid, 'Cleanup must target only this job group'
        if not alive:
            raise ProcessLookupError
        signals.append(sig)
        if sig == kill_signal:
            alive = False

    monkeypatch.setattr(suite.os, 'killpg', signal_group, raising=False)
    # A hashable process object matching the Popen interface.
    class Exited:
        pid = process.pid
        poll = staticmethod(process.poll)
        wait = staticmethod(process.wait)
    owned = Exited()
    runner = suite.Processes(termination_grace_seconds=0)
    if cancel:
        runner.active.add(owned)
        runner.cancel()
        assert not runner.active
        before = list(signals)
        runner.cancel()
        assert signals == before
    else:
        monkeypatch.setattr(suite.subprocess, 'Popen', lambda *args, **kwargs: owned)
        real_cleanup = suite.terminate_groups
        def tracked_cleanup(processes, timeout):
            assert owned in runner.active, 'Failed jobs must remain tracked until cleanup completes'
            real_cleanup(processes, timeout)
        monkeypatch.setattr(suite, 'terminate_groups', tracked_cleanup)
        with pytest.raises(subprocess.CalledProcessError, match='7'):
            runner.run(['failed-launcher'], tmp_path / 'failed.log')
        assert owned not in runner.active
    assert not alive and suite.signal.SIGTERM in signals and kill_signal in signals


def test_cleanup_does_not_kill_a_group_that_exits_gracefully(monkeypatch):
    process = SimpleNamespace(pid=4321, poll=lambda: 0, wait=lambda **kwargs: 0)
    signals = []
    def signal_group(pid, sig):
        if sig == 0:
            raise ProcessLookupError
        signals.append(sig)
    monkeypatch.setattr(suite.os, 'killpg', signal_group, raising=False)
    suite.terminate_groups([process], 0)
    assert signals == [suite.signal.SIGTERM]


def test_low_disk_during_suite_job_reaps_its_process_group(tmp_path, monkeypatch):
    runner = suite.Processes()
    runner.disk_paths = (tmp_path,)
    checks = []
    stopped = []
    def check(path):
        checks.append(path)
        if len(checks) > 1:
            raise OSError('disk reserve reached')
    class Running:
        def wait(self, timeout):
            raise subprocess.TimeoutExpired('test job', timeout)
    process = Running()
    monkeypatch.setattr(suite.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(suite, 'require_space', check)
    monkeypatch.setattr(suite, 'terminate_groups', lambda processes, grace: stopped.extend(processes))
    with pytest.raises(OSError, match='disk reserve'):
        runner.run(['test job'], tmp_path/'job.log')
    assert checks == [tmp_path, tmp_path] and stopped == [process] and not runner.active


@pytest.mark.skipif(os.name != 'posix', reason='Actual POSIX process-group signals require Linux/Unix')
def test_failed_launcher_reaps_a_real_term_ignoring_descendant(tmp_path):
    import psutil
    ready = tmp_path / 'child.pid'
    child = ('import os, signal, time; from pathlib import Path; '
             'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
             f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)')
    launcher = ('import subprocess, sys, time; from pathlib import Path\n'
                f'subprocess.Popen([sys.executable, "-c", {child!r}])\n'
                'for _ in range(500):\n'
                f'    if Path({str(ready)!r}).exists(): break\n'
                '    time.sleep(.01)\n'
                'sys.exit(7)\n')
    runner = suite.Processes(termination_grace_seconds=.15)
    try:
        with pytest.raises(subprocess.CalledProcessError):
            runner.run([sys.executable, '-c', launcher], tmp_path / 'real-process.log')
        assert ready.is_file()
        child_pid = int(ready.read_text())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                if psutil.Process(child_pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.01)
        else:
            pytest.fail('A descendant survived the failed launcher cleanup')
        assert not runner.active
    finally:
        if ready.exists():
            try:
                suite.os.kill(int(ready.read_text()), suite.signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != 'posix', reason='Production GPU locks use POSIX flock')
def test_uuid_aliases_contend_on_the_same_actual_file_lock(tmp_path):
    import fcntl
    identity = 'abcdef01-0000-0000-0000-000000000001'
    original = tmp_path / suite.gpu_lock_name(identity)
    alias = tmp_path / suite.gpu_lock_name('GPU-' + identity.upper())
    with original.open('a') as first, alias.open('a') as second:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(first, fcntl.LOCK_UN)
        fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_scheduler_releases_gpu_while_grade_runs_and_waits_before_large(tmp_path):
    plan = plan_for(tmp_path, ['--scales', *suite.SMALL, *suite.LARGE])
    lock = threading.Lock()
    active = set()
    completed_small = set()
    grading = threading.Event()
    large_full_graded = threading.Event()
    large_started = threading.Event()
    jobs = {j['scale']: j for q in plan['small_queues'] for j in q}
    jobs.update({j['scale']: j for j in plan['large']})
    class Fake:
        def run(self, command, log, cpu_only=False):
            scale = value(command, '--scale')
            if cpu_only:
                grading.set()
                assert large_started.wait(10), 'grading blocked GPU stage scheduling'
                if scale == 'deepseek-coder-33b' and value(command, '--rq4-strategy') == 'full':
                    large_full_graded.set()
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
    assert large_full_graded.is_set()
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
    monkeypatch.setattr(suite, 'resolve_gpu_uuids', lambda *args: pytest.fail('Dry-run must not query CUDA'))
    suite.main()
    assert json.loads(capsys.readouterr().out)['devices'] == ['0', '1', '2', '3']
    assert not dest.exists()


def test_execution_summary_shows_actual_gpu_ids_and_queue_order(capsys):
    args = suite.parser().parse_args(['--devices', '3,1,2,0'])
    plan = suite.build_suite(args, {}, list(range(64)), 1024)
    suite.print_execution_schedule(plan)
    output = capsys.readouterr().out
    for job in [j for queue in plan['small_queues'] for j in queue] + plan['large']:
        row = next(line for line in output.splitlines() if line.startswith(job['name'] + ' '))
        assert row.split('|')[1].strip() == ','.join(job['devices'])
    assert 'stage 2, exclusive 2-GPU job' in output


def test_execution_summary_explains_shared_gpu_queue(capsys):
    args = suite.parser().parse_args(['--devices', '0,1,2,3', '--max-parallel-small', '1'])
    plan = suite.build_suite(args, {}, list(range(64)), 1024)
    suite.print_execution_schedule(plan)
    output = capsys.readouterr().out
    for position, job in enumerate(plan['small_queues'][0], 1):
        row = next(line for line in output.splitlines() if line.startswith(job['name'] + ' '))
        assert row.split('|')[1].strip() == '0'
        assert f'GPU queue position {position}' in row


def test_default_suite_excludes_paused_models_and_keeps_all_explicit_choices(tmp_path):
    plan = plan_for(tmp_path)
    jobs = [j for q in plan['small_queues'] for j in q] + plan['large']
    assert {j['scale'] for j in jobs} == {'4b', 'gemma3-12b', 'deepseek-coder-1.3b', 'deepseek-coder-6.7b'}
    assert len(jobs) == 7
    assert not any(j['scale'] in {'1.7b', '8b', 'deepseek-coder-33b'} for j in jobs)
    assert all(j['devices'] == ['3', '1'] for j in plan['large'])
    full = plan_for(tmp_path, ['--scales', *suite.SMALL, *suite.LARGE])
    for job in full['large']:
        assert len(job['devices']) == h200.EXPERIMENTS[job['scale']].gpus


@pytest.mark.parametrize('scale', ['gemma3-12b', 'deepseek-coder-6.7b'])
def test_two_gpu_plan_keeps_train_and_evaluation_on_same_devices(tmp_path, scale):
    args = h200.parser().parse_args(['--scale', scale, '--devices', '0,1', '--output', str(tmp_path)])
    plan = h200.build_plan(args, {})
    assert plan['training_gpus'] == plan['evaluation_gpus'] == 2
    assert plan['rollout_tp'] == plan['evaluation_tp'] == 2
    assert plan['cuda_visible_devices'] == '0,1'
    assert value(plan['train'], '--num-gpus') == '2'
    assert value(plan['train'], '--rollout-tp-size') == '2'


def test_coder67_is_only_in_two_gpu_serial_stage(tmp_path):
    plan = plan_for(tmp_path, ['--devices', '0,1,2,3'])
    assert all(j['scale'] != 'deepseek-coder-6.7b' for q in plan['small_queues'] for j in q)
    coder = [j for j in plan['large'] if j['scale'] == 'deepseek-coder-6.7b']
    assert len(coder) == 1 and coder[0]['devices'] == ['0', '1']
    assert coder[0]['grade'] and coder[0]['rq4_strategy'] == 'full'
    args = h200.parser().parse_args(['--scale', 'deepseek-coder-6.7b', '--devices', '0'])
    with pytest.raises(ValueError, match='2 distinct visible GPUs'):
        h200.build_plan(args, {})
