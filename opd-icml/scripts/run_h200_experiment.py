"""Run one paired-full experiment, merge its final student, and evaluate on the same GPUs."""
import argparse
import hashlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_assets import read_models
from wandb_config import load_wandb_environment
from project_paths import project_absolute, project_relative, saved_path_matches
from package_results import archive_path, package_results
from cleanup_model_artifacts import finish_experiment, print_disk_space


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

# Throughput starting points; no change to sample counts, precision or diagnostics.
# Larger pairs retain offload because optimizer state is FP32.
PRESETS = {
    'deepseek-coder-1.3b': (2, 2, 64, 'off'),
    '1.7b': (2, 2, 64, 'off'),
    '4b': (2, 2, 64, 'off'),
    'deepseek-coder-6.7b': (1, 1, 32, 'on'),
    '8b': (2, 2, 32, 'on'),
    'gemma3-12b': (2, 2, 32, 'on'),
    'deepseek-coder-33b': (1, 1, 16, 'on'),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale', required=True, choices=EXPERIMENTS)
    p.add_argument('--task', choices=['math', 'code'], help='Default: DeepSeek code; Qwen/Gemma math')
    p.add_argument('--stage', choices=['all', 'train', 'eval', 'train-generate', 'generate', 'grade'], default='all')
    p.add_argument('--profile', choices=['throughput', 'conservative'], default='throughput')
    p.add_argument('--gpus', type=int, choices=[1, 2, 4], help='Override the preset GPU count for this student; TP follows it')
    p.add_argument('--save-freq', type=int, help='Resumable checkpoint every N rounds (passed to train.py)')
    p.add_argument('--resume', action='store_true', help='Continue training in an existing --output from its latest checkpoint')
    p.add_argument('--model-artifacts-root', type=Path, help='Keep the deletable weights (training checkpoints and merged model) under <root>/<run name>/ instead of inside the run; symlinks are left in the run directory')
    p.add_argument('--devices', help='Exactly one/four GPU IDs or UUIDs; otherwise use CUDA_VISIBLE_DEVICES. Ignored for CPU-only grade')
    p.add_argument('--output', type=Path, help='Pipeline root; contains training/, merged/, eval/, logs/')
    for name in ('student', 'teacher', 'cache-dir', 'model-dir', 'data', 'val-data'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--code-benchmarks', type=Path, default=Path('data/code_benchmarks'))
    p.add_argument('--math-benchmarks', type=Path, default=Path('data/benchmarks.jsonl'))
    p.add_argument('--micro-batch-size', type=int)
    p.add_argument('--teacher-micro-batch-size', type=int)
    p.add_argument('--optimizer-offload', choices=['on', 'off'])
    p.add_argument('--teacher-offload', choices=['on', 'off'])
    p.add_argument('--rollout-max-num-seqs', type=int)
    p.add_argument('--ray-num-cpus', type=int)
    p.add_argument('--ray-object-store-gib', type=float, default=4)
    p.add_argument('--gradient-cache-gib', type=float, default=64)
    p.add_argument('--cpu-ids', help='Linux CPU affinity, comma-separated IDs; inherited by all children')
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
    p.add_argument('--keep-model-files', action='store_true', help='Keep checkpoint/merged weights after successful evaluation; default cleans them')
    p.add_argument('--no-package', action='store_true', help='Skip the automatic experiment-data ZIP (no model files are ever packaged)')
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
    gpus = args.gpus or gpus
    task = args.task or spec.task
    if args.stage == 'grade' and task != 'code':
        raise ValueError('--stage grade is for code; math uses its existing combined scoring protocol')
    micro, teacher_micro, sequences, offload = (PRESETS[args.scale] if args.profile == 'throughput'
                                               else (1, 1, 16, 'on'))
    micro = args.micro_batch_size if args.micro_batch_size is not None else micro
    teacher_micro = args.teacher_micro_batch_size if args.teacher_micro_batch_size is not None else teacher_micro
    sequences = args.rollout_max_num_seqs if args.rollout_max_num_seqs is not None else sequences
    optimizer_offload = args.optimizer_offload or offload
    teacher_offload = args.teacher_offload or offload
    cpus = args.ray_num_cpus or max(1, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
    if args.ray_num_cpus is not None and args.ray_num_cpus < 1:
        raise ValueError('--ray-num-cpus must be positive')
    if args.stage in ('all', 'train', 'train-generate') and cpus < gpus + 1:
        raise ValueError(f'Training on {gpus} GPUs requires at least {gpus + 1} Ray CPUs '
                         '(one per GPU worker plus one TaskRunner)')
    if not 0 < args.ray_object_store_gib < float('inf') or not 0 <= args.gradient_cache_gib < float('inf'):
        raise ValueError('Ray memory must be positive and cache memory nonnegative, both finite')
    if task != 'code' and args.val_data:
        raise ValueError('--val-data applies only to code training')
    if min(args.eval_max_num_seqs, args.eval_batch_size, args.score_workers, args.code_workers) < 1:
        raise ValueError('Evaluation concurrency and worker counts must be positive')
    devices = '' if args.stage == 'grade' else select_devices(gpus, args.devices, environment)
    run = project_absolute(args.output or Path('runs/h200') / args.scale / f'{task}-paired-full')
    training, merged = run / 'training', run / 'merged'
    artifacts = project_absolute(args.model_artifacts_root) / run.name if args.model_artifacts_root else None
    checkpoints_root = artifacts / 'checkpoints' if artifacts else training / 'checkpoints'
    if artifacts:
        merged = artifacts / 'merged'
    ray_temp = Path(tempfile.gettempdir()) / ('opd-ray-' + hashlib.sha256(str(run).encode()).hexdigest()[:12])
    command = [sys.executable, 'train.py', '--scale', args.scale, '--task', task,
               '--experiment', 'paired-full', '--hardware', 'h200x1' if gpus == 1 else 'h200x4',
               '--num-gpus', str(gpus), '--rollout-tp-size', str(gpus),
               '--micro-batch-size', str(micro), '--teacher-micro-batch-size', str(teacher_micro),
               '--optimizer-offload', optimizer_offload, '--teacher-offload', teacher_offload,
               '--ray-num-cpus', str(cpus), '--ray-object-store-gib', str(args.ray_object_store_gib),
               '--ray-temp-dir', project_relative(ray_temp),
               '--rollout-max-num-seqs', str(sequences),
               '--rollout-gpu-memory-utilization', str(args.rollout_gpu_memory_utilization),
               '--learning-rate', str(args.learning_rate),
               '--parameter-token-gradients', args.parameter_token_gradients,
               '--code-workers', str(args.code_workers), '--output', project_relative(training)]
    if args.parameter_token_gradients == 'sampled':
        command += ['--parameter-token-samples', str(args.parameter_token_samples)]
    if args.parameter_token_group_size is not None:
        command += ['--parameter-token-group-size', str(args.parameter_token_group_size)]
    if args.save_freq is not None:
        command += ['--save-freq', str(args.save_freq)]
    if args.resume:
        command += ['--resume']
    if artifacts:
        command += ['--checkpoint-dir', project_relative(checkpoints_root)]
    for name in ('student', 'teacher', 'cache_dir', 'model_dir', 'data', 'val_data'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name.replace('_', '-'), project_relative(value)]
    prepare = []
    if task == 'code':
        if args.data is None or args.val_data is None:
            prepare.append([sys.executable, 'scripts/prepare_eurus_code.py'])
        evaluation = [sys.executable, 'scripts/evaluate_code.py',
            '--model-path', project_relative(merged), '--data-dir', project_relative(args.code_benchmarks),
            '--output-dir', project_relative(run / 'eval/code_pass8'), '--parallel-mode', 'tensor',
            '--prompt-style', 'chat', '--samples', '8', '--temperature', '1', '--top-p', '1',
            '--seed', '42', '--max-prompt-tokens', '4096', '--max-tokens', '8192',
            '--max-num-seqs', str(args.eval_max_num_seqs), '--batch-size', str(args.eval_batch_size),
            '--workers', str(args.score_workers)]
        prepare_eval = [sys.executable, 'scripts/prepare_code_benchmarks.py',
                        '--output-dir', project_relative(args.code_benchmarks)]
    else:
        if args.data is None:
            prepare.append([sys.executable, 'scripts/prepare_dapo14k.py',
                            '--output', 'data/dapo14k_train.parquet'])
        evaluation = [sys.executable, 'scripts/evaluate.py',
            '--model_path', project_relative(merged), '--input_file', project_relative(args.math_benchmarks),
            '--output_file', project_relative(run / 'eval/final_mean8.jsonl'), '--n', '8',
            '--temperature', '1', '--top_p', '1', '--seed', '42',
            '--max_prompt_tokens', '1024', '--max_tokens', '8192',
            '--max_num_seqs', str(args.eval_max_num_seqs), '--score_workers', str(args.score_workers)]
        prepare_eval = None
    checkpoint = checkpoints_root / 'global_step_<latest>/actor'
    merge = [sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
             '--local_dir', project_relative(checkpoint), '--target_dir', project_relative(merged), '--use_cpu_initialization']
    return dict(scale=args.scale, task=task, experiment='paired-full', gpu_count=gpus,
        training_gpus=gpus, evaluation_gpus=0 if args.stage == 'grade' else gpus,
        rollout_tp=gpus, evaluation_tp=None if args.stage == 'grade' else gpus,
        stage=args.stage, stage_gpu_count=0 if args.stage == 'grade' else gpus,
        keep_model_files=args.keep_model_files,
        results_zip=None if args.no_package else project_relative(archive_path(run, args.scale)),
        path_base='project_root', cuda_visible_devices=devices, output=project_relative(run), models=read_models(args.scale),
        checkpoints_dir=project_relative(checkpoints_root), merged_dir=project_relative(merged),
        model_artifacts=project_relative(artifacts) if artifacts else None,
        profile=args.profile, ray_cpus=cpus, gradient_cache_gib=args.gradient_cache_gib,
        prepare_training=prepare, prepare_evaluation=prepare_eval,
        train=command, merge=merge, evaluate=evaluation)


def child_environment(plan, environment):
    env = load_wandb_environment(environment)
    env['CUDA_VISIBLE_DEVICES'] = plan['cuda_visible_devices']
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    env['PYTHONUNBUFFERED'] = '1'
    env['RAY_ADDRESS'] = 'local'
    env['OPD_GRADIENT_CACHE_MAX_GIB'] = str(plan['gradient_cache_gib'])
    # Ray resources and BLAS threads are separate limits. Avoid nested CPU pools.
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT / 'runtime'), env.get('PYTHONPATH')]))
    return env


def final_checkpoint(checkpoints, gpus, expected_step):
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
    run = project_absolute(plan['output'])
    if args.stage == 'grade':
        run_command(plan['evaluate'] + ['--grade-only'], dict(env, CUDA_VISIBLE_DEVICES=''), run / 'logs/grade.log')
        return
    training_stage = args.stage in ('all', 'train', 'train-generate')
    if training_stage and not args.resume and run.exists() and any(run.iterdir()):
        raise ValueError(f'Choose an empty --output directory: {run}')
    # Use a fresh process so probing CUDA never initializes it in this orchestrator.
    probe = [sys.executable, '-c',
        'import torch; n=torch.cuda.device_count(); '
        f'assert n == {plan["gpu_count"]}, f"Expected {plan["gpu_count"]} visible GPUs, got {{n}}"; '
        'print("GPUs:", [torch.cuda.get_device_name(i) for i in range(n)])']
    run_command(probe, env, run / 'logs/devices.log')
    if training_stage:
        for number, command in enumerate(plan['prepare_training']):
            run_command(command, env, run / f'logs/prepare_training_{number}.log')
    if args.stage != 'train':
        if plan['task'] == 'code':
            # Existing benchmark manifests keep their chosen release/date window.
            from code_benchmarks import BENCHMARKS, read_manifest
            directory = project_absolute(args.code_benchmarks)
            if not all((directory / name / 'manifest.json').is_file() for name in BENCHMARKS):
                run_command(plan['prepare_evaluation'], env, run / 'logs/prepare_evaluation.log')
            for name in BENCHMARKS:
                read_manifest(directory, name)
        elif not project_absolute(args.math_benchmarks).is_file():
            raise ValueError(f'Missing math benchmark file: {args.math_benchmarks}')
    if training_stage:
        run_command(plan['train'], env, run / 'logs/train.log')
    if plan.get('model_artifacts'):
        # Convenience links only; the pipeline itself uses the real paths in the plan.
        for link, target in ((run / 'training/checkpoints', plan['checkpoints_dir']), (run / 'merged', plan['merged_dir'])):
            if not link.exists() and not link.is_symlink():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(project_absolute(target), target_is_directory=True)
    if args.stage == 'train':
        return
    expected = plan['resolved_training']['rounds_cap'] * plan['resolved_training']['updates_per_round']
    actor = final_checkpoint(project_absolute(plan['checkpoints_dir']), plan['gpu_count'], expected)
    merged = project_absolute(plan['merged_dir'])
    merge_marker = merged / 'h200_merge.json'
    identity = dict(scale=plan['scale'], task=plan['task'], checkpoint=project_relative(actor), gpu_count=plan['gpu_count'])
    if merged.exists():
        saved = json.loads(merge_marker.read_text(encoding='utf-8')) if merge_marker.is_file() else {}
        if (dict(saved, checkpoint=identity['checkpoint']) != identity
                or not saved_path_matches(saved.get('checkpoint', ''), actor)):
            raise ValueError(f'Unverified or incomplete merged model: {merged}; move it aside before retrying --stage eval')
    else:
        command = list(plan['merge'])
        command[command.index('--local_dir') + 1] = project_relative(actor)
        run_command(command, env, run / 'logs/merge.log')
        if not (merged / 'config.json').is_file() or not list(merged.glob('*.safetensors')):
            raise ValueError('Model merge did not produce a complete Hugging Face model')
        merge_marker.write_text(json.dumps(identity, indent=2) + '\n', encoding='utf-8')
    if plan['task'] == 'code':
        # Generate ALL benchmarks before CPU grading; suite scheduling can release
        # these GPUs immediately with train-generate/generate and grade separately.
        run_command(plan['evaluate'] + ['--generate-only'], env, run / 'logs/generate.log')
        if args.stage not in ('train-generate', 'generate'):
            run_command(plan['evaluate'] + ['--grade-only'], dict(env, CUDA_VISIBLE_DEVICES=''), run / 'logs/grade.log')
    else:
        run_command(plan['evaluate'], env, run / 'logs/evaluate.log')
    print(f'[DONE] Results: {project_relative(run / "eval")}', flush=True)


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.cpu_ids:
            selected = {int(item) for item in args.cpu_ids.split(',')}
            if not hasattr(os, 'sched_setaffinity') or not selected or not selected <= os.sched_getaffinity(0):
                raise ValueError('--cpu-ids must select available Linux CPUs')
            if not args.dry_run:
                os.sched_setaffinity(0, selected)
        plan = build_plan(args, os.environ)
        env = child_environment(plan, os.environ)
        # CPU grading needs neither training configuration nor any visible GPUs.
        if args.stage != 'grade':
            resolved = subprocess.run(plan['train'] + ['--dry-run'], cwd=ROOT, env=env,
                                      check=True, capture_output=True, text=True, encoding='utf-8')
            plan['resolved_training'] = json.loads(resolved.stdout)
        print(json.dumps(plan, indent=2), flush=True)
        if not args.dry_run:
            print_disk_space(plan['output'])
            execute(args, plan, env)
            if plan.get('model_artifacts'):
                print(f"[MODEL ARTIFACTS] weights kept under {plan['model_artifacts']} (delete that directory to reclaim space)", flush=True)
            else:
                finish_experiment(args, plan)
            if plan['results_zip']:
                package_results(plan['output'], plan['results_zip'], phase=args.stage)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(error.returncode)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()

