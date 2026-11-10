#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# Reuse the activated Python 3.10 evaluation environment; no nested venv and no
# second copy of PyTorch/vLLM. Fresh Python 3.10 users run setup_code_eval first.
python -c 'import sys,torch,importlib.metadata as m; assert sys.version_info[:2]==(3,10), "Activate Python 3.10"; assert torch.__version__=="2.6.0+cu124" and m.version("vllm")=="0.8.5.post1", "Run scripts/setup_code_eval.sh first"; assert not torch._C._GLIBCXX_USE_CXX11_ABI, "Prebuilt attention wheels require the standard Torch 2.6 cu124 ABI"'
python -m pip install --upgrade packaging ninja
python -m pip install --no-cache-dir --retries 10 --timeout 120 \
  -c configs/opd_training_constraints.txt -r runtime/requirements.txt \
  vllm evalplus scikit-learn math-verify matplotlib==3.9.4 lxml psutil pytest==8.4.2
python -m pip install --no-deps --no-cache-dir --retries 10 --timeout 120 \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' \
  'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1%2Bcu124torch2.6-cp38-abi3-linux_x86_64.whl'
python -m pip install --no-deps --no-cache-dir -e runtime
python -m pip check
python -c 'import torch,ray,tensordict,flash_attn,flashinfer; print("torch",torch.__version__,"CUDA",torch.version.cuda,"GPUs",torch.cuda.device_count(),"ray",ray.__version__); assert torch.cuda.is_available()'
