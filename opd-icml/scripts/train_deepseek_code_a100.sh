#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Activate code-eval310 before running this script. No packages are installed here.
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Edit these values or override them with environment variables.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"   # Questions generated per round.
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-$ROLLOUT_BATCH_SIZE}" # One OPD update uses the whole round.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"         # Questions per GPU forward/backward.
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}" # Concurrent generation sequences.
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
PARAMETER_TOKEN_SAMPLES="${PARAMETER_TOKEN_SAMPLES:-256}" # 64 high-KL + 192 remaining.
CODE_WORKERS="${CODE_WORKERS:-8}"                # CPU code judges, not GPU count.
STUDENT_PATH="${STUDENT_PATH:-models/deepseek-coder-1.3b-base}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/code/deepseek-coder-1.3b/paired-full-a100x1-sample256-eosv2-sh}"

# Current OPD protocol requires MINI_BATCH_SIZE == ROLLOUT_BATCH_SIZE.
# With 128 / 128 / 1 on one GPU, gradients accumulate over 128 micro-batches.
# Length limits, 106 rounds, diagnostics and final checkpoint follow the YAML.
# --dry-run prints the resolved settings without loading models or training.
exec "$PYTHON_BIN" train.py \
  --scale deepseek-coder-1.3b --task code \
  --experiment paired-full --hardware a100x1 \
  --student "$STUDENT_PATH" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --mini-batch-size "$MINI_BATCH_SIZE" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --rollout-max-num-seqs "$ROLLOUT_MAX_NUM_SEQS" \
  --rollout-gpu-memory-utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  --parameter-token-gradients sampled \
  --parameter-token-samples "$PARAMETER_TOKEN_SAMPLES" \
  --code-workers "$CODE_WORKERS" \
  --output "$OUTPUT_DIR" \
  "$@"
