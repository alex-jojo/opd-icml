#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3.10 -m venv .venv
source .venv/bin/activate
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"'

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install --retries 10 --timeout 120 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install --retries 10 --timeout 120 \
  -r "$ROOT/runtime/requirements.txt" \
  vllm==0.8.5.post1 \
  transformers==4.52.4 tokenizers==0.21.2 huggingface-hub==0.33.1 \
  tensordict==0.8.3 torchdata==0.11.0 \
  numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 \
  datasets==4.5.0 pyarrow==23.0.0 pandas==2.3.3 \
  ray==2.46.0 hydra-core==1.3.2 wandb==0.22.3 math-verify==0.9.0 \
  pytest==8.4.2

# Exact prebuilt wheels matching Python 3.10 + Torch 2.6 + CUDA 12.
# No compilation against the rented server's CUDA toolkit occurs here.
python -m pip install --no-cache-dir --retries 10 --timeout 120 \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -m pip install --no-cache-dir --retries 10 --timeout 120 \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1%2Bcu124torch2.6-cp38-abi3-linux_x86_64.whl"

python -m pip install -e "$ROOT/runtime" matplotlib==3.9.4 lxml psutil
python -m pip check
