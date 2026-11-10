"""Read-only live question progress for an already running code evaluation.

Reads newly appended JSONL bytes only. Never starts/stops generation, modifies
answers, loads model weights or uses a GPU. Use in a second terminal/tmux window.
"""
import argparse
import json
from pathlib import Path
import time
from tqdm import tqdm

NAMES = {'livecodebench': 'LiveCodeBench', 'mbppplus': 'MBPP+', 'humanevalplus': 'HumanEval+'}


class LineCounter:
    def __init__(self):
        self.identity = None
        self.offset = 0
        self.count = 0

    def read(self, path):
        try:
            with Path(path).open('rb') as stream:
                import os
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.identity, self.offset, self.count = identity, 0, 0
                stream.seek(self.offset)
                for block in iter(lambda: stream.read(1024*1024), b''):
                    # Unfinished final rows count only once their newline is flushed.
                    self.count += block.count(b'\n')
                    self.offset += len(block)
            return self.count
        except FileNotFoundError:
            return 0


class EvaluationProgress:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.run = json.loads((self.directory / 'run.json').read_text(encoding='utf8'))
        self.benchmarks = self.run['benchmarks']
        self.totals = {name: int(self.run['data'][name]['rows']) for name in self.benchmarks}
        if any(total < 1 for total in self.totals.values()):
            raise ValueError('Benchmark manifest must include a positive row count')
        self.counters = {}

    def count(self, path):
        return self.counters.setdefault(path, LineCounter()).read(path)

    def snapshot(self):
        result = {}
        for name in self.benchmarks:
            folder = self.directory / name
            total = self.totals[name]
            # The partial file is atomically renamed on completion; never add both.
            generated = self.count(folder / 'generations.jsonl')
            if not (folder / 'generations.jsonl').exists():
                generated = self.count(folder / 'generation.partial.jsonl')
                plan_path = folder / 'generation.plan.json'
                if plan_path.is_file():
                    plan = json.loads(plan_path.read_text(encoding='utf8'))
                    generated = self.count(folder / 'generation.seed.jsonl')
                    for index in range(plan['workers']):
                        shard = folder / 'generation_shards' / f"workers_{plan['workers']}" / f'shard_{index:03d}'
                        path = shard / 'generations.jsonl'
                        generated += self.count(path if path.exists() else shard / 'generation.partial.jsonl')
            judged = self.count(folder / 'livecodebench.results.jsonl') if name == 'livecodebench' else 0
            summary = folder / 'summary.json'
            if summary.is_file():
                saved = json.loads(summary.read_text(encoding='utf8'))
                judged = int(saved['problems'])
            if not 0 <= generated <= total or not 0 <= judged <= total:
                raise ValueError(f'Progress exceeds benchmark task count: {name}')
            stage = 'done' if judged == total else ('judging' if generated == total else
                    ('generating' if generated else 'waiting'))
            result[name] = dict(total=total, generated=generated, judged=judged, stage=stage)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--once', action='store_true', help='Print one JSON snapshot and exit')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    state = EvaluationProgress(args.output_dir)
    current = state.snapshot()
    if args.once:
        print(json.dumps(current, indent=2))
        return
    print('Live question progress (read-only; Ctrl+C closes only this view).', flush=True)
    print('EvalPlus live judging progress remains in the original run window; saved scores appear here on completion.', flush=True)
    total = sum(state.totals.values())
    bars = [tqdm(total=total, initial=sum(v['generated'] for v in current.values()),
                 desc='ALL | generated', position=0, unit='question', dynamic_ncols=True),
            tqdm(total=total, initial=sum(v['judged'] for v in current.values()),
                 desc='ALL | judged', position=1, unit='question', dynamic_ncols=True)]
    for index, name in enumerate(state.benchmarks, 2):
        bars.append(tqdm(total=state.totals[name], initial=current[name]['generated'],
                         desc=NAMES.get(name, name), position=index, unit='question', dynamic_ncols=True))
    try:
        while True:
            current = state.snapshot()
            values = [sum(v['generated'] for v in current.values()), sum(v['judged'] for v in current.values())]
            values += [current[name]['generated'] for name in state.benchmarks]
            for bar, value in zip(bars, values):
                bar.update(value-bar.n)
            for bar, name in zip(bars[2:], state.benchmarks):
                entry = current[name]
                bar.set_postfix_str(f"{entry['stage']}, judged {entry['judged']}/{entry['total']}")
            if all(entry['stage'] == 'done' for entry in current.values()):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        for bar in reversed(bars):
            bar.close()


if __name__ == '__main__':
    main()
