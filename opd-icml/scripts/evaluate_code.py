"""Generate eight samples per task, then run official code judges for pass@8."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from code_benchmarks import BENCHMARKS, EVALUATION_PROTOCOL as PROTOCOL, ROOT, digest, read_jsonl, read_manifest, save_json, validate_generations
from evaluation_prompts import CHAT_PROMPT_ENCODING, validate_chat_prompt_encoding
sys.path.insert(0, str(ROOT))
from project_paths import project_absolute, project_relative


def same_identity(saved, current):
    # Moving a checkpoint does not change its contents. All file hashes, prompt
    # encoding, datasets and sampling parameters still have to match exactly.
    def content(identity):
        return dict(identity, model={k: v for k, v in identity.get('model', {}).items() if k != 'path'})
    saved = content(saved)
    current = content(current)
    # One-way removal of MBPP+: keep existing answers only if every remaining
    # model/data/sampling/prompt identity still matches.
    if (saved.get('benchmarks') == ['livecodebench', 'mbppplus', 'humanevalplus']
            and current.get('benchmarks') == list(BENCHMARKS)
            and saved.get('protocol') == current.get('protocol') == PROTOCOL):
        saved['benchmarks'] = list(BENCHMARKS)
        saved['data'] = {key: value for key, value in saved.get('data', {}).items() if key != 'mbppplus'}
    return saved == current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, help='Local merged student or prepared Base/IT checkpoint')
    parser.add_argument('--benchmark', choices=['all', *BENCHMARKS], default='all')
    parser.add_argument('--data-dir', type=Path, default=Path('data/code_benchmarks'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8, help='>=8; default eight independent samples per task')
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--top-p', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt-style', choices=['chat', 'completion'], default='chat',
                        help='Use completion for Base code models without a chat template')
    parser.add_argument('--max-prompt-tokens', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--parallel-mode', choices=['tensor', 'data'], default='tensor',
                        help='data: one independent model replica per visible GPU, for models that fit one GPU')
    parser.add_argument('--timeout', type=int, default=6, help='Official LCB per-test timeout in seconds')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--grade-only', action='store_true', help='Use saved generations, no GPU/model loading')
    modes.add_argument('--generate-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for name in ('model_path', 'data_dir', 'output_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, project_absolute(getattr(args, name)))
    if args.samples < 8 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error('pass@8 requires >=8 sampled responses and positive temperature/top-p')
    if min(args.workers, args.batch_size, args.max_num_seqs, args.max_tokens, args.max_prompt_tokens, args.timeout) < 1:
        parser.error('Worker, batch, length and timeout settings must be positive')
    if not args.grade_only and args.model_path is None:
        parser.error('--model-path is required except for --grade-only')
    selected = BENCHMARKS if args.benchmark == 'all' else [args.benchmark]
    devices = None
    if args.parallel_mode == 'data' and not args.grade_only:
        from parallel_code_generation import visible_devices
        try:
            devices = visible_devices()
        except ValueError as error:
            parser.error(str(error))
    sampling = {key: getattr(args, key) for key in ('samples', 'temperature', 'top_p', 'seed', 'max_prompt_tokens', 'max_tokens')}
    if args.prompt_style != 'chat':
        sampling['prompt_style'] = args.prompt_style
    if args.dry_run:
        print(json.dumps(dict(benchmarks=list(selected), metric='pass@8', sampling=sampling,
                              parallel_mode=args.parallel_mode, generation_devices=devices,
                              data_dir=project_relative(args.data_dir), output_dir=project_relative(args.output_dir),
                              path_base='project_root',
                              generate=not args.grade_only, grade=not args.generate_only), indent=2))
        return
    manifests = {b: read_manifest(args.data_dir, b) for b in selected}
    marker = args.output_dir / 'run.json'
    identity = dict(protocol=PROTOCOL, benchmarks=list(selected), sampling=sampling,
                    data=manifests, metric='pass@8')
    if args.grade_only:
        previous = json.loads(marker.read_text(encoding='utf-8'))
        identity['model'] = previous['model']
        if 'prompt_encoding' in previous:
            identity['prompt_encoding'] = previous['prompt_encoding']
    else:
        model = args.model_path
        if not (model / 'config.json').is_file():
            parser.error('Expected a local model directory containing config.json')
        identity['model'] = dict(path=project_relative(model), sha256={p.name: digest(p) for p in sorted(model.iterdir()) if p.is_file()})
        if args.prompt_style == 'chat':
            identity['prompt_encoding'] = CHAT_PROMPT_ENCODING
    if marker.exists() and not same_identity(json.loads(marker.read_text(encoding='utf-8')), identity):
        raise ValueError('Code evaluation model/data/sampling/prompt encoding identity changed; use another --output-dir')
    if not args.grade_only and args.prompt_style == 'chat':
        # Also guard data-parallel resumes whose serial prefix becomes a seed.
        for benchmark in selected:
            for name in ('generations.jsonl', 'generation.partial.jsonl', 'generation.seed.jsonl'):
                cache = args.output_dir / benchmark / name
                if cache.exists():
                    validate_chat_prompt_encoding(list(read_jsonl(cache)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(marker, identity)
    worker = 'scripts/code_benchmark_worker.py'
    for number, benchmark in enumerate(selected, 1):
        print(f'\n[BENCHMARK {number}/{len(selected)}] {benchmark}', flush=True)
        base = [sys.executable, worker, '--benchmark', benchmark, '--data-dir', project_relative(args.data_dir),
                '--output-dir', project_relative(args.output_dir), '--prompt-style', args.prompt_style]
        for key in ('samples', 'temperature', 'top_p', 'seed', 'max_prompt_tokens', 'max_tokens', 'max_num_seqs', 'batch_size', 'workers', 'timeout'):
            base += ['--' + key.replace('_', '-'), str(getattr(args, key))]
        questions = list(read_jsonl(args.data_dir / benchmark / 'questions.jsonl'))
        if not args.grade_only:
            generation_command = base + ['--mode', 'generate', '--model-path', project_relative(args.model_path)]
            if args.parallel_mode == 'data':
                from parallel_code_generation import generate_parallel
                generate_parallel(generation_command, args.output_dir / benchmark, questions, args.samples, devices)
            else:
                subprocess.run(generation_command, cwd=ROOT, check=True)
        generations = list(read_jsonl(args.output_dir / benchmark / 'generations.jsonl'))
        validate_generations(questions, generations, args.samples)
        if not args.generate_only:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            subprocess.run(base + ['--mode', 'grade'], cwd=ROOT, env=env, check=True)
    if not args.generate_only:
        summaries = {b: json.loads((args.output_dir / b / 'summary.json').read_text(encoding='utf-8')) for b in selected}
        save_json(args.output_dir / 'summary.json', dict(metric='pass@8', benchmarks=summaries))
        for name, summary in summaries.items():
            print(f'{name}: pass@8={100 * summary["pass@8"]:.2f}% ({summary["problems"]} tasks)', flush=True)


if __name__ == '__main__':
    main()

