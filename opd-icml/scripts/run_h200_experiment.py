"""Run one paired-full experiment, merge its final student, and evaluate on the same GPUs."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_assets import read_models


@dataclass(frozen=True)
class Experiment:
    task: str
    gpus: int


# The threshold applies to the STUDENT, including the 8B -> 32B pair.
EXPERIMENTS = {
    'deepseek-coder-1.3b': Experiment('code', 1),
    'deepseek-coder-6.7b': Experiment('code', 1),
    'deepseek-coder-33b': Experiment('code', 4),
    '1.7b': Experiment('math', 1),
    '4b': Experiment('math', 1),
    '8b': Experiment('math', 4),
    'gemma3-12b': Experiment('math', 4),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale', required=True, choices=EXPERIMENTS)
    p.add_argument('--task', choices=['math', 'code'], help='Default: DeepSeek code; Qwen/Gemma math')
    p.add_argument('--stage', choices=['all', 'train', 'eval'], default='all')
    p.add_argument('--devices', help='Exactly one/four comma-separated GPU IDs or UUIDs; otherwise use CUDA_VISIBLE_DEVICES')
    p.add_argument('--output', type=Path, help='Pipeline root; contains training/, merged/, eval/, logs/')
    for name in ('student', 'teacher', 'cache-dir', 'model-dir', 'data', 'val-data'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--code-benchmarks', type=Path, default=ROOT / 'data/code_benchmarks')
    p.add_argument('--math-benchmarks', type=Path, default=ROOT / 'data/benchmarks.jsonl')
    p.add_argument('--micro-batch-size', type=int, default=1)
    p.add_argument('--rollout-max-num-seqs', type=int, default=16)
    p.add_argument('--rollout-gpu-memory-utilization', type=float, default=.30)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--parameter-token-gradients', choices=['sampled', 'exact-all', 'off'], default='sampled')
    p.add_argument('--parameter-token-samples', type=int, default=256)
    p.add_argument('--parameter-token-group-size', type=int)
    p.add_argument('--code-workers', type=int, default=8)
    p.add_argument('--eval-max-num-seqs', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=16, help='Code evaluation questions per generation batch')
    p.add_argument('--score-workers', type=int, default=8)
    p.add_argument('--dry-run', action='store_true', help='Print all commands and resolved training settings; no downloads/GPU/writes')
    return p


def select_devices(count, explicit, environment):
    raw = explicit if explicit is not None else environment.get('CUDA_VISIBLE_DEVICES')
    if raw is None:
        return ','.join(map(str, range(count)))
    devices = [item.strip() for item in raw.split(',')]
    if (any(not item or item == '-1' for item in devices)
            or len(set(devices)) != len(devices) or len(devices) < count):
        raise ValueError(f'This experiment requires {count} distinct visible GPUs')
    if explicit is not None and len(devices) != count:
        raise ValueError(f'--devices must select exactly {count} GPUs for this student')
    return ','.join(devices[:count])


def build_plan(args, environment):
    spec = EXPERIMENTS[args.scale]
    task = args.task or spec.task
    if task != 'code' and args.val_data:
        raise ValueError('--val-data applies only to code training')
    if min(args.eval_max_num_seqs, args.eval_batch_size, args.score_workers, args.code_workers) < 1:
        raise ValueError('Evaluation concurrency and worker counts must be positive')
    devices = select_devices(spec.gpus, args.devices, environment)
    run = (args.output or ROOT / 'runs/h200' / args.scale / f'{task}-paired-full').resolve()
    training, merged = run / 'training', run / 'merged'
    command = [sys.executable, str(ROOT / 'train.py'), '--scale', args.scale, '--task', task,
               '--experiment', 'paired-full', '--hardware', f'h200x{spec.gpus}',
               '--num-gpus', str(spec.gpus), '--rollout-tp-size', str(spec.gpus),
               '--micro-batch-size', str(args.micro_batch_size),
               '--rollout-max-num-seqs', str(args.rollout_max_num_seqs),
               '--rollout-gpu-memory-utilization', str(args.rollout_gpu_memory_utilization),
               '--learning-rate', str(args.learning_rate),
               '--parameter-token-gradients', args.parameter_token_gradients,
               '--code-workers', str(args.code_workers), '--output', str(training)]
    if args.parameter_token_gradients == 'sampled':
        command += ['--parameter-token-samples', str(args.parameter_token_samples)]
    if args.parameter_token_group_size is not None:
        command += ['--parameter-token-group-size', str(args.parameter_token_group_size)]
    for name in ('student', 'teacher', 'cache_dir', 'model_dir', 'data', 'val_data'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value.resolve())]
    prepare = []
    if task == 'code':
        if args.data is None or args.val_data is None:
            prepare.append([sys.executable, str(ROOT / 'scripts/prepare_eurus_code.py')])
        evaluation = [sys.executable, str(ROOT / 'scripts/evaluate_code.py'),
            '--model-path', str(merged), '--data-dir', str(args.code_benchmarks.resolve()),
            '--output-dir', str(run / 'eval/code_pass4'), '--parallel-mode', 'tensor',
            '--prompt-style', 'chat', '--samples', '4', '--temperature', '1', '--top-p', '1',
            '--seed', '42', '--max-prompt-tokens', '4096', '--max-tokens', '8192',
            '--max-num-seqs', str(args.eval_max_num_seqs), '--batch-size', str(args.eval_batch_size),
            '--workers', str(args.score_workers)]
        prepare_eval = [sys.executable, str(ROOT / 'scripts/prepare_code_benchmarks.py'),
                        '--output-dir', str(args.code_benchmarks.resolve())]
    else:
        if args.data is None:
            prepare.append([sys.executable, str(ROOT / 'scripts/prepare_dapo14k.py'),
                            '--output', str(ROOT / 'data/dapo14k_train.parquet')])
        evaluation = [sys.executable, str(ROOT / 'scripts/evaluate.py'),
            '--model_path', str(merged), '--input_file', str(args.math_benchmarks.resolve()),
            '--output_file', str(run / 'eval/final_mean8.jsonl'), '--n', '8',
            '--temperature', '1', '--top_p', '1', '--seed', '42',
            '--max_prompt_tokens', '1024', '--max_tokens', '8192',
            '--max_num_seqs', str(args.eval_max_num_seqs), '--score_workers', str(args.score_workers)]
        prepare_eval = None
    checkpoint = training / 'checkpoints/global_step_<latest>/actor'
    merge = [sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
             '--local_dir', str(checkpoint), '--target_dir', str(merged), '--use_cpu_initialization']
    return dict(scale=args.scale, task=task, experiment='paired-full', gpu_count=spec.gpus,
        training_gpus=spec.gpus, evaluation_gpus=spec.gpus, rollout_tp=spec.gpus, evaluation_tp=spec.gpus,
        cuda_visible_devices=devices, output=str(run), models=read_models(args.scale),
        prepare_training=prepare, prepare_evaluation=prepare_eval,
        train=command, merge=merge, evaluate=evaluation)


def child_environment(plan, environment):
    env = environment.copy()
    env['CUDA_VISIBLE_DEVICES'] = plan['cuda_visible_devices']
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT / 'runtime'), env.get('PYTHONPATH')]))
    env.setdefault('WANDB_MODE', 'offline')
    return env


def final_checkpoint(training, gpus, expected_step):
    checkpoints = training / 'checkpoints'
    marker = checkpoints / 'latest_checkpointed_iteration.txt'
    step = int(marker.read_text(encoding='utf-8').strip())
    if step != expected_step:
        raise ValueError(f'Expected the final step {expected_step}, found {step}; refusing partial-run evaluation')
    actor = checkpoints / f'global_step_{step}/actor'
    config = json.loads((actor / 'fsdp_config.json').read_text(encoding='utf-8'))
    if config.get('world_size') != gpus:
        raise ValueError('Checkpoint GPU count differs from this experiment')
    for rank in range(gpus):
        path = actor / f'model_world_size_{gpus}_rank_{rank}.pt'
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing/empty final model shard: {path}')
    if not (actor / 'huggingface/config.json').is_file():
        raise ValueError('Final checkpoint has no Hugging Face metadata')
    return actor


def run_command(command, env, log):
    print('[RUN] ' + shlex.join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\n[RUN] ' + shlex.join(command) + '\n')
        stream.flush()
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            result = process.wait()
        if result:
            raise subprocess.CalledProcessError(result, command)


def execute(args, plan, env):
    run = Path(plan['output'])
    if args.stage != 'eval' and run.exists() and any(run.iterdir()):
        raise ValueError(f'Choose an empty --output directory: {run}')
    # Use a fresh process so probing CUDA never initializes it in this orchestrator.
    probe = [sys.executable, '-c',
        'import torch; n=torch.cuda.device_count(); '
        f'assert n == {plan["gpu_count"]}, f"Expected {plan["gpu_count"]} visible GPUs, got {{n}}"; '
        'print("GPUs:", [torch.cuda.get_device_name(i) for i in range(n)])']
    run_command(probe, env, run / 'logs/devices.log')
    if args.stage != 'eval':
        for number, command in enumerate(plan['prepare_training']):
            run_command(command, env, run / f'logs/prepare_training_{number}.log')
    if args.stage != 'train':
        if plan['task'] == 'code':
            # Existing benchmark manifests keep their chosen release/date window.
            from code_benchmarks import BENCHMARKS, read_manifest
            directory = args.code_benchmarks.resolve()
            if not all((directory / name / 'manifest.json').is_file() for name in BENCHMARKS):
                run_command(plan['prepare_evaluation'], env, run / 'logs/prepare_evaluation.log')
            for name in BENCHMARKS:
                read_manifest(directory, name)
        elif not args.math_benchmarks.is_file():
            raise ValueError(f'Missing math benchmark file: {args.math_benchmarks}')
    if args.stage != 'eval':
        run_command(plan['train'], env, run / 'logs/train.log')
    if args.stage == 'train':
        return
    expected = plan['resolved_training']['rounds_cap'] * plan['resolved_training']['updates_per_round']
    actor = final_checkpoint(run / 'training', plan['gpu_count'], expected)
    merged = run / 'merged'
    merge_marker = merged / 'h200_merge.json'
    identity = dict(scale=plan['scale'], task=plan['task'], checkpoint=str(actor), gpu_count=plan['gpu_count'])
    if merged.exists():
        if not merge_marker.is_file() or json.loads(merge_marker.read_text(encoding='utf-8')) != identity:
            raise ValueError(f'Unverified or incomplete merged model: {merged}; move it aside before retrying --stage eval')
    else:
        command = list(plan['merge'])
        command[command.index('--local_dir') + 1] = str(actor)
        run_command(command, env, run / 'logs/merge.log')
        if not (merged / 'config.json').is_file() or not list(merged.glob('*.safetensors')):
            raise ValueError('Model merge did not produce a complete Hugging Face model')
        merge_marker.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    run_command(plan['evaluate'], env, run / 'logs/evaluate.log')
    print(f'[DONE] Results: {run / "eval"}', flush=True)


def main():
    p = parser()
    args = p.parse_args()
    try:
        plan = build_plan(args, os.environ)
        env = child_environment(plan, os.environ)
        # Resolve/validate the actual launcher before downloading anything.
        resolved = subprocess.run(plan['train'] + ['--dry-run'], cwd=ROOT, env=env,
                                  check=True, capture_output=True, text=True, encoding='utf-8')
        plan['resolved_training'] = json.loads(resolved.stdout)
        print(json.dumps(plan, indent=2), flush=True)
        if not args.dry_run:
            execute(args, plan, env)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
