#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -c 'import sys,platform; assert sys.version_info[:2] == (3,10), "Activate Python 3.10 first"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Linux x86_64 is required"'
python -m pip install --upgrade pip setuptools wheel
# PyPI is also required for CUDA dependency packages absent from the wheel index.
python -m pip install --retries 10 --timeout 120 -c configs/code_eval_constraints.txt \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -m pip install --retries 10 --timeout 120 -c configs/code_eval_constraints.txt \
  vllm transformers tokenizers huggingface-hub numpy scipy datasets pyarrow \
  evalplus tree-sitter tree-sitter-python stop-sequencer pytest==8.4.2
python -m pip check
python -c 'import torch,transformers,vllm; print("torch",torch.__version__,"CUDA",torch.version.cuda,"transformers",transformers.__version__,"vllm",vllm.__version__); assert torch.cuda.is_available(), "CUDA unavailable; check driver/GPU access"'
