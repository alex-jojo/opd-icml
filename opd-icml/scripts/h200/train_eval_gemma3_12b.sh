#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# 2 H200 GPU(s) for BOTH training and evaluation. Activate the project environment first.
exec "${PYTHON_BIN:-python}" scripts/run_h200_experiment.py --scale gemma3-12b "$@"
