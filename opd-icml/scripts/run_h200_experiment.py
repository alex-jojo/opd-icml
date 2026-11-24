"""Run one paired-full experiment, merge its final student, and evaluate on the same GPUs."""
import argparse
from contextlib import nullcontext
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
from cleanup_model_artifacts import finish_experiment, print_disk_space, temporary_early_export, register_external_models
import rq4_results
from cpu_resources import resolve_cpu_resources, bounded_thread_environment


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
    'gemma3-12b': Experiment('math', 2),
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
    p.add_argument('--rq4-strategy', choices=rq4_results.STRATEGIES,
                   help='Matched FULL, early-stop or reuse strategy with task-specific scores and timing')
    p.add_argument('--task', choices=['math', 'code'], help='Default: DeepSeek code; Qwen/Gemma math')
    p.add_argument('--stage', choices=['all', 'train', 'eval', 'train-generate', 'generate', 'grade'], default='all')
    p.add_argument('--profile', choices=['throughput', 'conservative'], default='throughput')
    p.add_argument('--gpus', type=int, choices=[1, 2, 4], help='Override the preset GPU count for this student; TP follows it')
    p.add_argument('--save-freq', type=int, help='Resumable checkpoint every N rounds (passed to train.py)')
    p.add_argument('--resume', action='store_true', help='Continue training in an existing --output from its latest checkpoint')
    p.add_argument('--model-artifacts-root', type=Path, help='Keep the deletable weights (training checkpoints and merged model) under <root>/<run name>/ instead of inside the run; symlinks are left in the run directory')
    p.add_argument('--devices', help='Exactly one/two/four GPU IDs or UUIDs; otherwise use CUDA_VISIBLE_DEVICES. Ignored for CPU-only grade')
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
    p.add_argument('--parameter-token-gradients', choices=['sampled', 'exact-all', 'off'],
                   help='Default: sampled for new runs, saved mode on resume')
    p.add_argument('--parameter-token-samples', type=int, help='Default: 64 for new runs, saved count on resume')
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
    gpus = args.gpus or spec.gpus
    task = args.task or spec.task
    strategy = args.rq4_strategy
    label = f'rq4-{strategy}' if strategy else 'paired-full'
    run = project_absolute(args.output or Path('runs/h200') / args.scale / f'{task}-{label}')
    legacy_full = False
    if args.resume or args.stage in ('eval', 'generate', 'grade'):
        marker = run/'rq4_training_time.json'
        if marker.is_file():
            record = rq4_results.read(marker)
            saved = record['strategy']
            if record['scale'] != args.scale or saved not in rq4_results.STRATEGIES or strategy not in (None, saved):
                raise ValueError('RQ4 model or strategy differs from the saved training run')
            strategy = saved
            legacy_full = strategy == 'full' and not record.get('shared_early_stop', False)
        elif strategy == 'full':
            # Runs predating the timing record cannot supply a first-stable checkpoint.
            legacy_full = True
    elif strategy is None and (task == 'code' or project_absolute(args.math_benchmarks) == ROOT/'data/benchmarks.jsonl'):
        strategy = 'full'
    if args.stage == 'grade' and task != 'code':
        raise ValueError('--stage grade is for code; math uses its existing combined scoring protocol')
    micro, teacher_micro, sequences, offload = (PRESETS[args.scale] if args.profile == 'throughput'
                                               else (1, 1, 16, 'on'))
    micro = args.micro_batch_size if args.micro_batch_size is not None else micro
    teacher_micro = args.teacher_micro_batch_size if args.teacher_micro_batch_size is not None else teacher_micro
    sequences = args.rollout_max_num_seqs if args.rollout_max_num_seqs is not None else sequences
    optimizer_offload = args.optimizer_offload or offload
    teacher_offload = args.teacher_offload or offload
    cpu = resolve_cpu_resources(gpus, args.ray_num_cpus, args.cpu_ids,
                                training=args.stage in ('all', 'train', 'train-generate'))
    cpus = cpu['ray_cpus']
    if not 0 < args.ray_object_store_gib < float('inf') or not 0 <= args.gradient_cache_gib < float('inf'):
        raise ValueError('Ray memory must be positive and cache memory nonnegative, both finite')
    if task != 'code' and args.val_data:
        raise ValueError('--val-data applies only to code training')
    if min(args.eval_max_num_seqs, args.eval_batch_size, args.score_workers, args.code_workers) < 1:
        raise ValueError('Evaluation concurrency and worker counts must be positive')
    devices = '' if args.stage == 'grade' else select_devices(gpus, args.devices, environment)
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
               '--code-workers', str(args.code_workers), '--output', project_relative(training)]
    if strategy and not legacy_full:
        command += ['--rq4-strategy', strategy]
    mode = args.parameter_token_gradients
    if mode is None and not args.resume:
        mode = 'sampled'
    if mode is not None:
        command += ['--parameter-token-gradients', mode]
    samples = args.parameter_token_samples
    if samples is None and mode == 'sampled' and not args.resume:
        samples = 64
    if samples is not None and (args.resume or mode == 'sampled'):
        command += ['--parameter-token-samples', str(samples)]
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
    evaluate_base = None
    if (strategy in (None, 'full') and args.stage != 'train' and args.student is None and task == spec.task
            and (task == 'code' or project_absolute(args.math_benchmarks) == ROOT/'data/benchmarks.jsonl')):
        evaluate_base = [sys.executable, 'scripts/evaluate_base.py', '--scale', args.scale, '--no-package',
                         '--output', project_relative(run/'eval/base'),
                         '--eval-max-num-seqs', str(args.eval_max_num_seqs),
                         '--score-workers', str(args.score_workers)]
        if task == 'code':
            evaluate_base += ['--code-benchmarks', project_relative(args.code_benchmarks),
                              '--eval-batch-size', str(args.eval_batch_size)]
        else:
            evaluate_base += ['--math-benchmarks', project_relative(args.math_benchmarks)]
        for flag, path in [('--cache-dir', args.cache_dir), ('--model-dir', args.model_dir)]:
            if path is not None:
                evaluate_base += [flag, project_relative(path)]
    checkpoint = checkpoints_root / 'global_step_<latest>/actor'
    merge = [sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
             '--local_dir', project_relative(checkpoint), '--target_dir', project_relative(merged), '--use_cpu_initialization']
    return dict(scale=args.scale, task=task, experiment='paired-full', rq4_strategy=strategy,
        shared_early_stop=strategy == 'full' and not legacy_full, gpu_count=gpus,
        training_gpus=gpus, evaluation_gpus=0 if args.stage == 'grade' else gpus,
        rollout_tp=gpus, evaluation_tp=None if args.stage == 'grade' else gpus,
        stage=args.stage, stage_gpu_count=0 if args.stage == 'grade' else gpus,
        keep_model_files=args.keep_model_files,
        results_zip=None if args.no_package else project_relative(archive_path(run, args.scale)),
        path_base='project_root', cuda_visible_devices=devices, output=project_relative(run), models=read_models(args.scale),
        checkpoints_dir=project_relative(checkpoints_root), merged_dir=project_relative(merged),
        model_artifacts=project_relative(artifacts) if artifacts else None,
        profile=args.profile, ray_cpus=cpus, cpu_ids=cpu['cpu_ids'], cpu_resources=cpu,
        gradient_cache_gib=args.gradient_cache_gib,
        prepare_training=prepare, prepare_evaluation=prepare_eval,
        train=command, merge=merge, evaluate=evaluation, evaluate_base=evaluate_base)


def child_environment(plan, environment):
    env = bounded_thread_environment(load_wandb_environment(environment))
    env['CUDA_VISIBLE_DEVICES'] = plan['cuda_visible_devices']
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    env['PYTHONUNBUFFERED'] = '1'
    env['RAY_ADDRESS'] = 'local'
    env['OPD_GRADIENT_CACHE_MAX_GIB'] = str(plan['gradient_cache_gib'])
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT / 'runtime'), env.get('PYTHONPATH')]))
    return env


def final_checkpoint(checkpoints, gpus, expected_step):
    marker = checkpoints / 'latest_checkpointed_iteration.txt'
    step = int(marker.read_text(encoding='utf-8').strip())
    if step != expected_step:
        raise ValueError(f'Expected the final step {expected_step}, found {step}; refusing partial-run evaluation')
    return checked_actor(checkpoints / f'global_step_{step}/actor', gpus)


def checked_actor(actor, gpus):
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


def early_code_command(plan, run):
    command = list(plan['evaluate'])
    command[command.index('--output-dir') + 1] = project_relative(run/'eval/early-stop/code_pass8')
    return command


def grade_code(plan, run, env):
    strategy = plan['rq4_strategy']
    cpu = dict(env, CUDA_VISIBLE_DEVICES='')
    if strategy:
        # The generation stage validates the actual endpoint and writes this only
        # after both sets of generations complete. CPU regrading needs no weights.
        receipt = rq4_results.read(run/'training/training_records/rq4/code_endpoints.json')
        if (receipt['scale'], receipt['strategy']) != (plan['scale'], strategy):
            raise ValueError('Code grading endpoints do not match this experiment')
        selection = receipt['early_stop']
        if bool(selection) != plan['shared_early_stop']:
            raise ValueError('Code grading endpoints do not match the early-stop mode')
        if selection and not selection['uses_full_endpoint']:
            run_command(early_code_command(plan, run) + ['--grade-only'], cpu, run/'logs/grade_early_stop.log')
        base_generated = receipt.get('evaluate_base', False)
    else:
        # Older runs have no RQ4 receipt. An explicit Base generation can still be graded.
        base_generated = (run/'eval/base/code_pass8/run.json').is_file()
    if base_generated:
        if not plan['evaluate_base']:
            raise ValueError('CPU grading must include the scheduled Base evaluation')
        run_command(plan['evaluate_base'] + ['--grade-only'], cpu, run/'logs/grade_base.log')
    run_command(plan['evaluate'] + ['--grade-only'], cpu, run/'logs/grade.log')
    if strategy:
        arm = rq4_results.save_arm(run, plan['scale'], strategy, receipt['endpoint'], 'code')
        if selection:
            rq4_results.save_shared_arm(run, arm, selection)


def evaluate_early_checkpoint(plan, run, checkpoints, env):
    snapshot = checkpoints/'early_stop'
    actor = checked_actor(snapshot/'actor', plan['gpu_count'])
    # The model shards remain available for retries. Release the temporary HF
    # export before merging the final FULL model.
    with temporary_early_export(snapshot) as export:
        merged = project_relative(export)
        merge = list(plan['merge'])
        merge[merge.index('--local_dir') + 1] = project_relative(actor)
        merge[merge.index('--target_dir') + 1] = merged
        run_command(merge, env, run/'logs/merge_early_stop.log')
        if plan['task'] == 'code':
            evaluation = early_code_command(plan, run) + ['--generate-only']
            evaluation[evaluation.index('--model-path') + 1] = merged
        else:
            evaluation = list(plan['evaluate'])
            evaluation[evaluation.index('--model_path') + 1] = merged
            evaluation[evaluation.index('--output_file') + 1] = project_relative(run/'eval/early-stop/final_mean8.jsonl')
        run_command(evaluation, env, run/'logs/evaluate_early_stop.log')


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
    rq4 = run/'training/training_records/rq4'
    strategy = plan['rq4_strategy']
    if args.stage == 'grade':
        grade_code(plan, run, env)
        return
    training_stage = args.stage in ('all', 'train', 'train-generate')
    if training_stage and not args.resume and run.exists() and any(run.iterdir()):
        raise ValueError(f'Choose an empty --output directory: {run}')
    if training_stage and plan.get('model_artifacts'):
        register_external_models(plan, allow_unowned_resume=args.resume)
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
        timer = (rq4_results.training_timer(run, plan['scale'], strategy, args.resume, plan['shared_early_stop'])
                 if strategy else nullcontext())
        with timer:
            run_command(plan['train'], env, run / 'logs/train.log')
    if plan.get('model_artifacts'):
        # Convenience links only; the pipeline itself uses the real paths in the plan.
        for link, target in ((run / 'training/checkpoints', plan['checkpoints_dir']), (run / 'merged', plan['merged_dir'])):
            if not link.exists() and not link.is_symlink():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(project_absolute(target), target_is_directory=True)
    if args.stage == 'train':
        return
    expected = plan['resolved_training']['rounds_cap']
    if strategy:
        end = rq4_results.endpoint(project_absolute(plan['checkpoints_dir']), rq4, strategy,
                                   expected, plan['resolved_training']['updates_per_round'])
        expected = end['rollout_rounds']
    actor = final_checkpoint(project_absolute(plan['checkpoints_dir']), plan['gpu_count'], expected)
    selection = None
    if plan['shared_early_stop']:
        checkpoints = project_absolute(plan['checkpoints_dir'])
        selection = rq4_results.shared_selection(checkpoints, rq4, expected)
        if not selection['uses_full_endpoint']:
            evaluate_early_checkpoint(plan, run, checkpoints, env)
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
        if plan['evaluate_base']:
            run_command(plan['evaluate_base'] + ['--generate-only'], env, run/'logs/generate_base.log')
        if strategy:
            rq4_results.save_json(rq4/'code_endpoints.json', dict(scale=plan['scale'], strategy=strategy,
                                  endpoint=end, early_stop=selection, evaluate_base=bool(plan['evaluate_base'])))
        if args.stage not in ('train-generate', 'generate'):
            grade_code(plan, run, env)
    else:
        run_command(plan['evaluate'], env, run / 'logs/evaluate.log')
        if plan['evaluate_base']:
            run_command(plan['evaluate_base'], env, run / 'logs/evaluate_base.log')
        if strategy:
            full_or_independent = rq4_results.save_arm(run, plan['scale'], strategy, end)
            if selection is not None:
                rq4_results.save_shared_arm(run, full_or_independent, selection)
    print(f'[DONE] Results: {project_relative(run / "eval")}', flush=True)


def main():
    p = parser()
    args = p.parse_args()
    try:
        plan = build_plan(args, os.environ)
        if not args.dry_run:
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, set(plan['cpu_ids']))
            elif args.cpu_ids is not None:
                raise ValueError('--cpu-ids requires Linux CPU affinity support')
        env = child_environment(plan, os.environ)
        if args.stage in ('eval', 'generate'):
            # Evaluation needs just the endpoint budget, even on a one-CPU allocation.
            from train import training_config, paired_rq4_overrides
            _, config = training_config(ROOT / 'configs' / args.scale, plan['experiment'], plan['task'])
            paired_rq4_overrides(config, plan['rq4_strategy'])
            plan['resolved_training'] = dict(rounds_cap=config['trainer']['opd_pause_after_rollout_rounds'],
                updates_per_round=config['actor_rollout_ref']['actor']['policy_loss']['jump_reuse_updates'])
        elif args.stage != 'grade':
            resolved = subprocess.run(plan['train'] + ['--dry-run'], cwd=ROOT, env=env,
                                      check=True, capture_output=True, text=True, encoding='utf-8')
            plan['resolved_training'] = json.loads(resolved.stdout)
        print(json.dumps(plan, indent=2), flush=True)
        if not args.dry_run:
            print_disk_space(plan['output'])
            execute(args, plan, env)
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
