#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Use one already activated Python 3.10 environment throughout.
bash scripts/setup_code_eval.sh
bash scripts/setup_opd_training.sh
bash scripts/setup_code_benchmarks.sh
echo "[READY] H200 training and evaluation dependencies installed."
