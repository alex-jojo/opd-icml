"""Launch one paper experiment using its recorded Hydra configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from model_assets import model_profiles, prepare_models

ROOT = Path(__file__).resolve().parent


def training_config(profile, experiment, task):
    name = ('code-' if task == 'code' else '') + experiment
    config = json.loads((profile / (name + '.yaml')).read_text(encoding='utf-8'))
    if task == 'math':
        return name, config
    base = json.loads((profile / (config['defaults'][0] + '.yaml')).read_text(encoding='utf-8'))
    def merge(target, overlay):
        for key, value in overlay.items():
            if key == 'defaults':
                continue
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
    merge(base, config)
    return name, base


def validate_code_dataset(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    required = {'data_source', 'prompt', 'ability', 'reward_model', 'extra_info'}
    if not required.issubset(parquet.schema_arrow.names) or parquet.metadata.num_rows == 0:
        raise ValueError(f'Missing/empty code dataset fields in {path}')
    rows = parquet.read(columns=['ability', 'data_source']).to_pylist()
    if any(row['ability'] != 'code' or row['data_source'] not in {'apps', 'codecontests', 'codeforces', 'taco'} for row in rows):
        raise ValueError(f'Expected code-only Eurus data in {path}; run scripts/prepare_eurus_code.py')


def apply_hardware_profile(cfg, hardware):
    """Keep recorded experiment files intact; return explicit Hydra overrides."""
    if hardware is None:
        return {}
    if hardware != 'l40x4':
        raise ValueError(f'Unknown hardware profile: {hardware}')
    policy = cfg['actor_rollout_ref']['actor']['policy_loss']
    replication = policy.get('replication17', {})
    if not replication.get('enabled'):
        raise ValueError('l40x4 requires the distributed replication protocol')
    if replication.get('heavy_steps'):
        raise ValueError('This profile has single-GPU disposable shadow probes; l40x4 cannot run them. '
                         'Use paired-full for the four-GPU RQ1 run; probes were not silently disabled.')
    overrides = {
        'trainer.n_gpus_per_node': 4,
        'actor_rollout_ref.actor.policy_loss.replication17.distributed_world_size': 4,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 4,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        'actor_rollout_ref.ref.fsdp_config.use_torch_compile': False,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.gpu_memory_utilization': .3,
        'actor_rollout_ref.rollout.max_num_seqs': 16,
    }
    for path, value in overrides.items():
        target = cfg
        parts = path.split('.')
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=model_profiles(), required=True)
    parser.add_argument("--experiment", choices=["paired-full", "original-full", "early-stop", "reuse"])
    parser.add_argument("--task", choices=["math", "code"], default="math")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--val-data", type=Path, help="Code validation parquet; defaults to Eurus code validation split")
    parser.add_argument("--sandbox-url", help="Use a Sandbox Fusion /run_code endpoint for code correctness")
    parser.add_argument("--code-workers", type=int, default=8, help="Concurrent code submissions to judge")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--student", type=Path, help="Local student model directory")
    parser.add_argument("--teacher", type=Path, help="Local teacher model directory")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare both model checkpoints without starting training")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    parser.add_argument("--model-dir", type=Path, help="Root for Gemma text-only exports; defaults to models/")
    parser.add_argument("--dry-run", action="store_true", help="Show the configuration without downloading models or starting training")
    parser.add_argument("--parameter-token-gradients", choices=["exact-all", "off"],
                        help="Override full per-token parameter-gradient recording (gradient-study profiles only)")
    parser.add_argument("--hardware", choices=["l40x4"], help="Four-L40 memory-conscious runtime overrides")
    args = parser.parse_args()
    if args.code_workers < 1:
        parser.error("--code-workers must be positive")
    if args.task != 'code' and (args.val_data or args.sandbox_url):
        parser.error("--val-data and --sandbox-url apply to --task code")
    profile = ROOT / "configs" / args.scale
    models = json.loads((profile / "models.json").read_text(encoding="utf-8"))
    if args.prepare_only:
        if args.dry_run:
            print(json.dumps(dict(scale=args.scale, models=models, training=False), indent=2))
            return
        try:
            paths = prepare_models(args.scale, args.student, args.teacher, args.cache_dir, args.model_dir)
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(dict(scale=args.scale, training=False, **paths), indent=2))
        return
    if args.experiment is None:
        parser.error("--experiment is required unless --prepare-only is used")
    config_name, cfg = training_config(profile, args.experiment, args.task)
    try:
        hardware_overrides = apply_hardware_profile(cfg, args.hardware)
    except ValueError as error:
        parser.error(str(error))
    study = cfg['actor_rollout_ref']['actor']['policy_loss'].get('gradient_study', {})
    if args.parameter_token_gradients:
        if not study.get('enabled', False):
            parser.error('--parameter-token-gradients requires a gradient-study profile, such as paired-full')
        study['parameter_tokens'] = dict(enabled=args.parameter_token_gradients == 'exact-all', mode='exact_all')
    default_run = ROOT / 'runs' / args.scale / args.experiment if args.task == 'math' else ROOT / 'runs/code' / args.scale / args.experiment
    output = (args.output or default_run).resolve()
    data = (args.data or ROOT / ('data/eurus_code_train.parquet' if args.task == 'code' else 'data/dapo14k_train.parquet')).resolve()
    validation = (args.val_data or ROOT / 'data/eurus_code_validation.parquet').resolve() if args.task == 'code' else data
    command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-dir", str(profile),
               "--config-name", config_name, "hydra.run.dir=" + str(output / "hydra")]
    command += ['++' + key + '=' + json.dumps(value) for key, value in hardware_overrides.items()]
    if args.parameter_token_gradients:
        prefix = '++actor_rollout_ref.actor.policy_loss.gradient_study.parameter_tokens.'
        command += [prefix + 'enabled=' + str(study['parameter_tokens']['enabled']).lower(), prefix + 'mode=exact_all']
    print(json.dumps(dict(experiment=args.experiment, scale=args.scale, task=args.task, gpu_count=cfg["trainer"]["n_gpus_per_node"],
                          hardware_profile=args.hardware,
                          micro_batch_size_per_gpu=cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'],
                          rollout_tensor_parallel_size=cfg['actor_rollout_ref']['rollout']['tensor_model_parallel_size'],
                          rollout_batch=cfg["data"]["train_batch_size"], rounds_cap=cfg["trainer"]["opd_pause_after_rollout_rounds"],
                          updates_per_round=cfg["actor_rollout_ref"]["actor"]["policy_loss"]["jump_reuse_updates"],
                          output=str(output), data=str(data), validation_data=str(validation),
                          max_prompt_tokens=cfg['data']['max_prompt_length'], max_response_tokens=cfg['data']['max_response_length'],
                          code_judge=('sandbox' if args.sandbox_url else 'local') if args.task == 'code' else None,
                          parameter_token_gradients=study.get('parameter_tokens', {'enabled': False}),
                          models=models, command=shlex.join(command)), indent=2), flush=True)
    if args.dry_run:
        return
    if not data.is_file():
        parser.error(f"Missing training parquet: {data}. See README for the preparation command.")
    if args.task == 'code':
        if not validation.is_file():
            parser.error(f'Missing code validation parquet: {validation}; run scripts/prepare_eurus_code.py')
        if validation == data:
            parser.error('Code validation data must use a separate split from training')
        try:
            validate_code_dataset(data)
            validate_code_dataset(validation)
        except ValueError as error:
            parser.error(str(error))
        if not args.sandbox_url and os.name == 'nt':
            parser.error('The local code judge requires a Linux/Unix training server, or pass --sandbox-url')
    if output.exists() and any(output.iterdir()):
        parser.error(f"Choose an empty output directory: {output}")
    env = os.environ.copy()
    try:
        paths = prepare_models(args.scale, args.student, args.teacher, args.cache_dir, args.model_dir)
    except ValueError as error:
        parser.error(str(error))
    for role, path in paths.items():
        env["OPD_" + role.upper()] = path
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    env.update(OPD_DATA=str(data), OPD_RUN=str(output), OPD_PROJECT_ROOT=str(ROOT),
               OPD_VALIDATION_DATA=str(validation), OPD_CODE_WORKERS=str(args.code_workers),
               OPD_CODE_JUDGE_BACKEND='sandbox' if args.sandbox_url else 'local',
               OPD_CODE_SANDBOX_URL=args.sandbox_url or '', TOKENIZERS_PARALLELISM="true",
               VLLM_WORKER_MULTIPROC_METHOD="spawn",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "runtime"), env.get("PYTHONPATH")])))
    env.setdefault("WANDB_MODE", "offline")
    subprocess.run(command, cwd=output, env=env, check=True)


if __name__ == "__main__":
    main()
