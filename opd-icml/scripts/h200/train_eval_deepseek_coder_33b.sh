#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# 4 H200 GPU(s) for BOTH training and evaluation. Activate the project environment first.
exec "${PYTHON_BIN:-python}" scripts/run_h200_experiment.py --scale deepseek-coder-33b "$@"
