"""Measured training costs and five-benchmark scores for the matched RQ4 runs."""
from contextlib import contextmanager
import json
import time

from code_benchmarks import save_json
from project_paths import project_absolute


SCALES = ('1.7b', '4b')
STRATEGIES = ('full', 'early-stop', 'reuse')
BENCHMARKS = ('math500', 'hmmt25_feb', 'hmmt26_feb', 'aime26', 'minerva')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@contextmanager
def training_timer(run, scale, strategy, resume):
    # train.py requires its training/ directory to be empty at first launch.
    path = run/'rq4_training_time.json'
    record = read(path) if path.exists() else dict(scale=scale, strategy=strategy, attempts=[], complete_history=not resume)
    if (record['scale'], record['strategy']) != (scale, strategy):
        raise ValueError('RQ4 model or strategy differs from the saved run')
    attempt = dict(status='running')
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


def save_arm(run, scale, strategy, end):
    timing = read(run/'rq4_training_time.json')
    attempts = timing['attempts']
    if (timing['scale'], timing['strategy']) != (scale, strategy) or attempts[-1]['status'] != 'complete':
        raise ValueError('RQ4 evaluation and completed training strategy must match')
    complete = timing['complete_history'] and all('seconds' in attempt for attempt in attempts)
    wall = sum(attempt['seconds'] for attempt in attempts) if complete else None
    scores = read(run/'eval/final_mean8.summary.json')
    summary = dict(scale=scale, strategy=strategy, **end, training_wall_seconds=wall,
                   **five_benchmark_scores(scores))
    save_json(run/'training/training_records/rq4/summary.json', summary)


def save_comparison(plan):
    groups = {}
    for queue in plan['small_queues']:
        for job in queue:
            strategy = job.get('rq4_strategy')
            if strategy:
                folder = project_absolute(job['output'])/'training/training_records/rq4'
                row = read(folder/'summary.json')
                if (row['scale'], row['strategy']) != (job['scale'], strategy):
                    raise ValueError('RQ4 summary does not match its scheduled experiment')
                arms = groups.setdefault(job['scale'], {})
                arms[strategy] = row
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
            row['five_benchmark_change_pp'] = 100*(row['five_benchmark_mean_at_8']-reference['five_benchmark_mean_at_8'])
    if groups:
        save_json(project_absolute(plan['output'])/'rq4_comparison.json', groups)
