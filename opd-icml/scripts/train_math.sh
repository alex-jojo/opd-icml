#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Activate the existing training environment before running this script.
PYTHON_BIN="${PYTHON_BIN:-python}"
SCALE="${SCALE:-1.7b}"                    # 1.7b | 4b | 8b | gemma3-12b
EXPERIMENT="${EXPERIMENT:-paired-full}"  # paired-full | original-full | early-stop | reuse
HARDWARE="${HARDWARE:-a100x1}"            # a100x1 | l40x4 | recorded (keep YAML hardware)

# auto inherits the selected experiment's YAML. Change to a number to override.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-auto}" # paired-full: 128; original-full: 160.
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-auto}"       # One full rollout per actor call.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"       # Per GPU, per forward/backward.
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"

# auto uses sampled diagnostics for paired-full, inherits YAML for other arms.
PARAMETER_TOKEN_GRADIENTS="${PARAMETER_TOKEN_GRADIENTS:-auto}" # auto | sampled | exact-all | off | inherit
PARAMETER_TOKEN_SAMPLES="${PARAMETER_TOKEN_SAMPLES:-256}"    # sampled: 64 high-KL + 192 remaining.
STUDENT_PATH="${STUDENT_PATH:-}"   # Empty: download/reuse the profile's pinned student.
TEACHER_PATH="${TEACHER_PATH:-}"   # Empty: download/reuse the profile's pinned teacher.
DATA_PATH="${DATA_PATH:-}"         # Empty: data/dapo14k_train.parquet.
OUTPUT_DIR="${OUTPUT_DIR:-}"       # Empty: runs/math/<scale>/<experiment>-<hardware>-<mode>-sh.

case "$SCALE" in 1.7b|4b|8b|gemma3-12b) ;; *) printf 'Unsupported math SCALE: %s\n' "$SCALE" >&2; exit 2 ;; esac
case "$EXPERIMENT" in paired-full|original-full|early-stop|reuse) ;; *) printf 'Unsupported EXPERIMENT: %s\n' "$EXPERIMENT" >&2; exit 2 ;; esac

# Read the recorded batch and GPU count without loading models or training.
PROFILE_DEFAULTS="$("$PYTHON_BIN" - "$SCALE" "$EXPERIMENT" <<'PY'
import json
from pathlib import Path
import sys
cfg=json.loads((Path('configs')/sys.argv[1]/(sys.argv[2]+'.yaml')).read_text(encoding='utf-8'))
print(cfg['data']['train_batch_size'])
print(cfg['trainer']['n_gpus_per_node'])
PY
)"
mapfile -t PROFILE_VALUES <<< "$PROFILE_DEFAULTS"
if [[ "$ROLLOUT_BATCH_SIZE" == auto ]]; then ROLLOUT_BATCH_SIZE="${PROFILE_VALUES[0]}"; fi
if [[ "$MINI_BATCH_SIZE" == auto ]]; then MINI_BATCH_SIZE="$ROLLOUT_BATCH_SIZE"; fi
if [[ "$PARAMETER_TOKEN_GRADIENTS" == auto ]]; then
  if [[ "$EXPERIMENT" == paired-full ]]; then PARAMETER_TOKEN_GRADIENTS=sampled; else PARAMETER_TOKEN_GRADIENTS=inherit; fi
fi

hardware_args=()
case "$HARDWARE" in
  a100x1) GPU_COUNT=1; hardware_args=(--hardware a100x1) ;;
  l40x4) GPU_COUNT=4; hardware_args=(--hardware l40x4) ;;
  recorded) GPU_COUNT="${PROFILE_VALUES[1]}" ;;
  *) printf 'Unsupported HARDWARE: %s\n' "$HARDWARE" >&2; exit 2 ;;
esac
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=0
  for ((GPU_INDEX=1; GPU_INDEX<GPU_COUNT; GPU_INDEX++)); do CUDA_VISIBLE_DEVICES+=",$GPU_INDEX"; done
fi
export CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn

gradient_args=()
case "$PARAMETER_TOKEN_GRADIENTS" in
  inherit) ;;
  sampled) gradient_args=(--parameter-token-gradients sampled --parameter-token-samples "$PARAMETER_TOKEN_SAMPLES") ;;
  exact-all|off) gradient_args=(--parameter-token-gradients "$PARAMETER_TOKEN_GRADIENTS") ;;
  *) printf 'Unsupported PARAMETER_TOKEN_GRADIENTS: %s\n' "$PARAMETER_TOKEN_GRADIENTS" >&2; exit 2 ;;
esac
asset_args=()
if [[ -n "$STUDENT_PATH" ]]; then asset_args+=(--student "$STUDENT_PATH"); fi
if [[ -n "$TEACHER_PATH" ]]; then asset_args+=(--teacher "$TEACHER_PATH"); fi
if [[ -n "$DATA_PATH" ]]; then asset_args+=(--data "$DATA_PATH"); fi
if [[ -z "$OUTPUT_DIR" ]]; then OUTPUT_DIR="runs/math/$SCALE/$EXPERIMENT-$HARDWARE-$PARAMETER_TOKEN_GRADIENTS-sh"; fi

# Budgets, model revisions, lengths, validation, and checkpoints follow YAML.
# 128/128/1 on one GPU: accumulate 128 micro-batches before one optimizer update.
# Original-full and legacy early-stop/reuse arms keep their 160 = 128 train + 32 probe protocol.
# Unsupported combinations (e.g. four-GPU disposable probes) fail explicitly in train.py.
exec "$PYTHON_BIN" train.py \
  --scale "$SCALE" --task math --experiment "$EXPERIMENT" \
  "${hardware_args[@]}" "${asset_args[@]}" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --mini-batch-size "$MINI_BATCH_SIZE" \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --rollout-max-num-seqs "$ROLLOUT_MAX_NUM_SEQS" \
  --rollout-gpu-memory-utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  "${gradient_args[@]}" --output "$OUTPUT_DIR" "$@"
