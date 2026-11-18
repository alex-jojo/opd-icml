"""Measured training costs and task-specific evaluation scores for matched RQ4 runs."""
from contextlib import contextmanager
import json
import time

from code_benchmarks import BENCHMARKS as CODE_BENCHMARKS, save_json
from project_paths import project_absolute


STRATEGIES = ('full', 'early-stop', 'reuse')
BENCHMARKS = ('math500', 'hmmt25_feb', 'hmmt26_feb', 'aime26', 'minerva')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@contextmanager
def training_timer(run, scale, strategy, resume, shared_early_stop=False):
    # train.py requires its training/ directory to be empty at first launch.
    path = run/'rq4_training_time.json'
    if path.exists():
        record = read(path)
    else:
        record = dict(scale=scale, strategy=strategy, attempts=[], complete_history=not resume)
        if shared_early_stop:
            record['shared_early_stop'] = True
    if (record['scale'], record['strategy']) != (scale, strategy):
        raise ValueError('RQ4 model or strategy differs from the saved run')
    attempt = dict(status='running', started_at_unix=time.time())
    record['attempts'].append(attempt)
    save_json(path, record)
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        attempt['status'] = 'interrupted'
        raise
    else:
        attempt['status'] = 'complete'
    finally:
        attempt['seconds'] = time.perf_counter() - started
        attempt['ended_at_unix'] = time.time()
        save_json(path, record)


def endpoint(checkpoints, directory, strategy, rounds_cap, reuse):
    step = int((checkpoints/'latest_checkpointed_iteration.txt').read_text())
    counts = read(checkpoints/f'global_step_{step}'/'opd_optimizer_updates.json')
    if counts['rollout_round'] != step or counts['optimizer_updates_done'] != step*reuse or not 1 <= step <= rounds_cap:
        raise ValueError('RQ4 checkpoint counters disagree with the training budget')
    reason = 'budget'
    if strategy == 'early-stop':
        window = directory/f'early_stop_{step:03d}.json'
        if window.is_file():
            decision = read(window)
            if decision['step'] == step and decision['stable']:
                reason = 'stable'
    if step < rounds_cap and reason != 'stable':
        raise ValueError('RQ4 checkpoint is neither the budget endpoint nor a confirmed stability stop')
    return dict(rollout_rounds=step, optimizer_updates=counts['optimizer_updates_done'], stop_reason=reason)


def five_benchmark_scores(scores):
    means = {}
    for name in BENCHMARKS:
        entry = scores[name]
        if entry['problems'] <= 0 or entry['samples'] != 8*entry['problems'] or not 0 <= entry['mean_at_8'] <= 1:
            raise ValueError(f'Incomplete Mean@8 results for {name}')
        means[name] = entry['mean_at_8']
    return dict(mean_at_8=means, five_benchmark_mean_at_8=sum(means.values())/len(means))


def evaluation_scores(directory, task):
    if task == 'math':
        return five_benchmark_scores(read(directory/'final_mean8.summary.json'))
    summary = read(directory/'code_pass8/summary.json')
    if summary['metric'] != 'pass@8' or set(summary['benchmarks']) != set(CODE_BENCHMARKS):
        raise ValueError('Expected both code pass@8 benchmarks')
    scores = {}
    for name, row in summary['benchmarks'].items():
        if row['problems'] <= 0 or row['samples'] != 8*row['problems'] or not 0 <= row['pass@8'] <= 1:
            raise ValueError(f'Incomplete pass@8 results for {name}')
        scores[name] = row['pass@8']
    return dict(pass_at_8=scores, two_benchmark_mean_pass_at_8=sum(scores.values())/len(scores))


def save_arm(run, scale, strategy, end, task='math'):
    timing = read(run/'rq4_training_time.json')
    attempts = timing['attempts']
    if (timing['scale'], timing['strategy']) != (scale, strategy) or attempts[-1]['status'] != 'complete':
        raise ValueError('RQ4 evaluation and completed training strategy must match')
    complete = timing['complete_history'] and all('seconds' in attempt for attempt in attempts)
    wall = sum(attempt['seconds'] for attempt in attempts) if complete else None
    scores = evaluation_scores(run/'eval', task)
    summary = dict(scale=scale, strategy=strategy, **end, training_wall_seconds=wall,
                   task=task, **scores)
    save_json(run/'training/training_records/rq4/summary.json', summary)
    return summary


def shared_selection(checkpoints, decisions, final_step):
    selected = read(checkpoints/'early_stop/selection.json')
    step = selected['step']
    if not 1 <= step <= final_step or selected['optimizer_updates'] != step or selected['uses_full_endpoint'] != (step == final_step):
        raise ValueError('Early-stop checkpoint does not match the FULL trajectory')
    # Ignore uncommitted future windows left by a rolled-back attempt. Check
    # every scheduled observation through the selection (or through the cap).
    first_stable = None
    interval = selected['interval']
    for update in range(interval, step + 1, interval):
        decision = read(decisions/f'early_stop_{update:03d}.json')
        if decision['step'] != update:
            raise ValueError('Early-stop decision has the wrong update number')
        if decision['stable']:
            first_stable = update
            break
    reason = selected['stop_reason']
    if not ((reason == 'stable' and first_stable == step) or
            (reason == 'budget' and first_stable is None and step == final_step)):
        raise ValueError('Early-stop selection disagrees with the first stable window')
    return selected


def prefix_wall_seconds(timing, selected_at):
    if not timing['complete_history']:
        return None
    elapsed = 0.
    for attempt in timing['attempts']:
        if 'seconds' not in attempt:
            return None
        if attempt.get('started_at_unix', float('inf')) <= selected_at <= attempt.get('ended_at_unix', 0):
            return elapsed + (selected_at - attempt['started_at_unix'])
        elapsed += attempt['seconds']
    return None


def save_shared_arm(run, full, selection):
    task = full.get('task', 'math')
    at_budget = selection['uses_full_endpoint']
    directory = 'eval' if at_budget else 'eval/early-stop'
    scores = evaluation_scores(run/directory, task)
    if task == 'code':
        responses = dict(responses_files={name: f'{directory}/code_pass8/{name}/generations.jsonl'
                                          for name in CODE_BENCHMARKS})
    else:
        responses = dict(responses_file=f'{directory}/final_mean8.jsonl')
    wall = (full['training_wall_seconds'] if at_budget else
            prefix_wall_seconds(read(run/'rq4_training_time.json'), selection['selected_at_unix']))
    summary = dict(scale=full['scale'], strategy='early-stop', source_strategy='full',
                   rollout_rounds=selection['step'], optimizer_updates=selection['optimizer_updates'],
                   stop_reason=selection['stop_reason'], training_wall_seconds=wall, uses_full_endpoint=at_budget,
                   timing_basis='full_endpoint' if at_budget else 'full_prefix_through_checkpoint_save',
                   task=task, **responses, **scores)
    save_json(run/'training/training_records/rq4/early_stop_summary.json', summary)


def save_comparison(plan):
    groups = {}
    for queue in [*plan['small_queues'], plan.get('large', [])]:
        for job in queue:
            strategy = job.get('rq4_strategy')
            if strategy:
                folder = project_absolute(job['output'])/'training/training_records/rq4'
                row = read(folder/'summary.json')
                if (row['scale'], row['strategy']) != (job['scale'], strategy):
                    raise ValueError('RQ4 summary does not match its scheduled experiment')
                arms = groups.setdefault(job['scale'], {})
                arms[strategy] = row
                if job.get('shared_early_stop'):
                    shared = read(folder/'early_stop_summary.json')
                    if (shared['scale'], shared['strategy'], shared['source_strategy']) != (job['scale'], 'early-stop', 'full'):
                        raise ValueError('Shared early-stop result does not match its FULL run')
                    arms['early-stop'] = shared
                if job['scale'] == '4b' and strategy == 'full':
                    base = project_absolute(job['output'])/'eval/base/summary.json'
                    if base.is_file():  # Older completed suites predate the Base evaluation.
                        result = read(base)
                        if (result['scale'], result['strategy']) != ('4b', 'base'):
                            raise ValueError('Base summary does not match the scheduled 4B experiment')
                        arms['base'] = {key: result[key] for key in ('scale', 'strategy', 'model',
                                        'rollout_rounds', 'optimizer_updates', 'training_wall_seconds')}
                        arms['base'].update(scoring_mode='strict_last_box',
                                            mean_at_8=result['strict_last_box']['mean_at_8'],
                                            five_benchmark_mean_at_8=result['strict_last_box']['five_benchmark_mean_at_8'])
    for arms in groups.values():
        if set(arms) - {'base'} != set(STRATEGIES):
            raise ValueError('RQ4 comparison requires every matched arm')
        reference = arms['full']
        for row in arms.values():
            wall, base = row['training_wall_seconds'], reference['training_wall_seconds']
            row['training_time_saved_fraction'] = 1-wall/base if wall is not None and base else None
            if reference.get('task') == 'code':
                row['two_benchmark_change_pp'] = 100*(row['two_benchmark_mean_pass_at_8']-reference['two_benchmark_mean_pass_at_8'])
            else:
                row['five_benchmark_change_pp'] = 100*(row['five_benchmark_mean_at_8']-reference['five_benchmark_mean_at_8'])
    if groups:
        save_json(project_absolute(plan['output'])/'rq4_comparison.json', groups)
