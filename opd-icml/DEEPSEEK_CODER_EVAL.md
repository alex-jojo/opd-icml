# DeepSeek-Coder 1.3B Base：从零运行 code pass@4

以下在 Linux x86_64 服务器执行，项目已上传到服务器。`cd` 后应能看到 `scripts/evaluate_code.py`、`configs/` 和 `third_party/`。使用此版本的整个项目（包含新增 `setup_code_eval.sh` 和 `--prompt-style completion`），不要只复制旧测评脚本。

这是独立推理测评环境，不启动 OPD 训练，不加载 teacher，不计算梯度，不需要 Eurus 训练集。先用 1 张 L40 运行。模型固定为官方 `deepseek-ai/deepseek-coder-1.3b-base`，revision 为 `c919139c3a9b4070729c8b2cca4847ab29ca8d94`。

## 1. 进入项目并建立环境

如果服务器没有 conda，先执行一次（已有 conda 则跳过）：

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o /tmp/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

随后执行：

```bash
cd /path/to/opd-icml
nvidia-smi
conda create -n code-eval310 -c conda-forge --override-channels python=3.10 -y
conda activate code-eval310
bash scripts/setup_code_eval.sh
```

若已按之前步骤配置好训练环境，可激活原环境后执行 `bash scripts/setup_code_benchmarks.sh`，但独立新环境更容易排查依赖。安装入口固定 PyTorch 2.6.0/cu124、vLLM 0.8.5.post1、Transformers 4.52.4、EvalPlus 0.3.1；不安装 Ray/VERL、训练专用 FlashAttention wheel。服务器需要兼容 CUDA 12.4 的 NVIDIA 驱动，建议 R550 或更新驱动。

## 2. 下载固定版本模型

```bash
mkdir -p models
huggingface-cli download deepseek-ai/deepseek-coder-1.3b-base \
  --revision c919139c3a9b4070729c8b2cca4847ab29ca8d94 \
  --local-dir models/deepseek-coder-1.3b-base
```

模型约 2.7 GB；已经下载的文件会复用。无需为 Base 模型创建或修改 chat template。

## 3. 准备三个 benchmark

```bash
python scripts/prepare_code_benchmarks.py --benchmark all --lcb-release release_v6
```

准备 HumanEval+ 164 题、MBPP+ 378 题及 LiveCodeBench release_v6，使用项目固定版本和校验值。LCB 原始文件合计约 4.49 GB，需额外留出处理后数据与输出空间。此步骤需要访问 Hugging Face 和 EvalPlus GitHub release，复用已完成且校验通过的数据。

## 4. 运行 pass@4

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p results

python scripts/evaluate_code.py \
  --model-path models/deepseek-coder-1.3b-base \
  --benchmark all \
  --prompt-style completion \
  --samples 4 --temperature 1.0 --top-p 1.0 --seed 42 \
  --max-prompt-tokens 8192 --max-tokens 8192 \
  --max-num-seqs 8 --batch-size 2 --workers 8 \
  --output-dir results/deepseek-coder-1.3b-base-pass4
```

每题采样 4 次，按通过任意一次的题目比例计算 pass@4。每个 benchmark 先生成并退出 GPU 子进程，再运行 CPU 官方判题。HumanEval+/MBPP+ 要同时通过 base 和 plus tests；LCB 使用 public+private tests。

这里为提示和回答各预留 8192 token，总计 16384，对应该模型配置的上下文长度；并发限制为 8 条生成序列。

Base 模型用 `--prompt-style completion`：HumanEval+/MBPP+ 直接续写官方代码前缀，判题时补回该前缀；LCB 使用题目和打开的 Python 代码块作为前缀，生成在关闭代码块时停止。输入使用模型 tokenizer 的 BOS 设置；不套 chat template。提示方式记录在 run manifest 和回答中。这是本项目的固定三 benchmark 协议，不声称复现原 DeepSeek 发布的分数或其 few-shot 提示。

只先检查 HumanEval+ 时，将 `--benchmark all` 改成 `--benchmark humanevalplus`，并使用另一个 `--output-dir`；正式跑三个 benchmark 使用上面的新目录。输出目录记录模型、题集、采样与 prompt style 身份，改变设置必须换目录。提示超长会明确报错，不静默截断/丢题。

## 5. 查看结果

```bash
python -m json.tool results/deepseek-coder-1.3b-base-pass4/summary.json
```

每项的 `pass@4` 为 0–1 的比例，乘以 100 是百分数。子目录分别保留回答、判题详情和 summary。中断后用同一命令续跑；只重判时可去掉 `--model-path` 并添加 `--grade-only`，其余 benchmark、prompt style 和采样参数保持一致。

尚未在此电脑上下载完整模型或执行 GPU 测评。Base 提示、代码拼接、pass@4 与既有 chat 协议已进行 CPU 回归验证；实际分数和速度以服务器结果为准。
