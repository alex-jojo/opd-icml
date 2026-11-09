"""Launch one paper experiment using its recorded Hydra configuration."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["1.7b", "4b"], required=True)
    parser.add_argument("--experiment", choices=["paired-full", "original-full", "early-stop", "reuse"], required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/dapo14k_train.parquet")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--student", type=Path, help="Local student model directory")
    parser.add_argument("--teacher", type=Path, help="Local teacher model directory")
    parser.add_argument("--dry-run", action="store_true", help="Show the configuration without downloading models or starting training")
    args = parser.parse_args()
    profile = ROOT / "configs" / args.scale
    cfg = json.loads((profile / (args.experiment + ".yaml")).read_text())
    models = json.loads((profile / "models.json").read_text())
    output = (args.output or ROOT / "runs" / args.scale / args.experiment).resolve()
    data = args.data.resolve()
    command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-dir", str(profile),
               "--config-name", args.experiment, "hydra.run.dir=" + str(output / "hydra")]
    print(json.dumps(dict(experiment=args.experiment, scale=args.scale, gpu_count=cfg["trainer"]["n_gpus_per_node"],
                          rollout_batch=cfg["data"]["train_batch_size"], rounds_cap=cfg["trainer"]["opd_pause_after_rollout_rounds"],
                          updates_per_round=cfg["actor_rollout_ref"]["actor"]["policy_loss"]["jump_reuse_updates"],
                          output=str(output), data=str(data), models=models, command=shlex.join(command)), indent=2), flush=True)
    if args.dry_run:
        return
    if not data.is_file():
        parser.error(f"Missing training parquet: {data}. See README for the preparation command.")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Choose an empty output directory: {output}")
    env = os.environ.copy()
    for role in ["student", "teacher"]:
        local = getattr(args, role)
        if local:
            if not (local / "config.json").is_file():
                parser.error(f"Missing {role} config.json in {local}")
            path = local.resolve()
        else:
            from huggingface_hub import snapshot_download
            path = snapshot_download(models[role]["repo"], revision=models[role]["revision"])
        env["OPD_" + role.upper()] = str(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    env.update(OPD_DATA=str(data), OPD_RUN=str(output), TOKENIZERS_PARALLELISM="true",
               VLLM_WORKER_MULTIPROC_METHOD="spawn",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "runtime"), env.get("PYTHONPATH")])))
    env.setdefault("WANDB_MODE", "offline")
    subprocess.run(command, cwd=output, env=env, check=True)


if __name__ == "__main__":
    main()
