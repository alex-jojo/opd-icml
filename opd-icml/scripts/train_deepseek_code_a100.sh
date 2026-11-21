#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Activate code-eval310 before running this script. No packages are installed here.
PYTHON_BIN="${PYTHON_BIN:-python}"
HARDWARE="${HARDWARE:-a100x1}"    # Memory preset; explicit NUM_GPUS overrides its GPU count.
NUM_GPUS="${NUM_GPUS:-1}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-$NUM_GPUS}"
if [[ ! "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then printf 'NUM_GPUS must be positive\n' >&2; exit 2; fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=0
  for ((GPU_INDEX=1; GPU_INDEX<NUM_GPUS; GPU_INDEX++)); do CUDA_VISIBLE_DEVICES+=",$GPU_INDEX"; done
fi
export CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Edit these values or override them with environment variables.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"   # Questions generated per round.
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-$ROLLOUT_BATCH_SIZE}" # One OPD update uses the whole round.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"         # Questions per GPU forward/backward.
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}" # Concurrent generation sequences.
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
PARAMETER_TOKEN_SAMPLES="${PARAMETER_TOKEN_SAMPLES:-64}" # 16 high-KL + 48 remaining.
CODE_WORKERS="${CODE_WORKERS:-8}"                # CPU code judges, not GPU count.
STUDENT_PATH="${STUDENT_PATH:-models/deepseek-coder-1.3b-base}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/code/deepseek-coder-1.3b/paired-full-$HARDWARE-gpus$NUM_GPUS-tp$ROLLOUT_TP_SIZE-sample$PARAMETER_TOKEN_SAMPLES-eosv2-perf}"

# Current OPD protocol requires MINI_BATCH_SIZE == ROLLOUT_BATCH_SIZE.
# With 128 / 128 / 1 on one GPU, gradients accumulate over 128 micro-batches.
# Length limits, 106 rounds, diagnostics and final checkpoint follow the YAML.
# --dry-run prints the resolved settings without loading models or training.
exec "$PYTHON_BIN" train.py \
  --scale deepseek-coder-1.3b --task code \
  --experiment paired-full --hardware "$HARDWARE" \
  --num-gpus "$NUM_GPUS" --rollout-tp-size "$ROLLOUT_TP_SIZE" \
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
