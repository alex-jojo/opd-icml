# DeepSeek-Coder 1.3B Base：从零运行 code pass@4

以下在 Linux x86_64 服务器执行，项目已上传到服务器。`cd` 后应能看到 `scripts/evaluate_code.py`、`configs/` 和 `third_party/`。使用此版本的整个项目（包含新增 `setup_code_eval.sh` 和 `--prompt-style completion`），不要只复制旧测评脚本。

这是独立推理测评环境，不启动 OPD 训练，不加载 teacher，不计算梯度，不需要 Eurus 训练集。下面按用户的四张 L40S 运行，默认推荐经过小样验证的 TP=4、较大批量。模型固定为官方 `deepseek-ai/deepseek-coder-1.3b-base`，revision 为 `c919139c3a9b4070729c8b2cca4847ab29ca8d94`。

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p results

python scripts/evaluate_code.py \
  --model-path models/deepseek-coder-1.3b-base \
  --benchmark all \
  --prompt-style completion \
  --parallel-mode tensor \
  --samples 4 --temperature 1.0 --top-p 1.0 --seed 42 \
  --max-prompt-tokens 8192 --max-tokens 8192 \
  --max-num-seqs 16 --batch-size 8 --workers 32 \
  --output-dir results/deepseek-coder-1.3b-base-pass4-tp4-batch8
```

每题采样 4 次，按通过任意一次的题目比例计算 pass@4。每个 benchmark 先生成并退出 GPU 子进程，再运行 CPU 官方判题。HumanEval+/MBPP+ 要同时通过 base 和 plus tests；LCB 使用 public+private tests。

这里为提示和回答各预留 8192 token，总计 16384，对应该模型配置的上下文长度；并发限制为 16 条生成序列。一次提交 8 道题、每题 4 个回答，引擎持续调度这些采样请求。`--workers 32` 是 CPU 判题进程数，不是 GPU 数。

Base 模型用 `--prompt-style completion`：HumanEval+/MBPP+ 直接续写官方代码前缀，判题时补回该前缀；LCB 使用题目和打开的 Python 代码块作为前缀，生成在关闭代码块时停止。输入使用模型 tokenizer 的 BOS 设置；不套 chat template。提示方式记录在 run manifest 和回答中。这是本项目的固定三 benchmark 协议，不声称复现原 DeepSeek 发布的分数或其 few-shot 提示。

只先检查 HumanEval+ 时，将 `--benchmark all` 改成 `--benchmark humanevalplus`，并使用另一个 `--output-dir`；正式跑三个 benchmark 使用上面的新目录。输出目录记录模型、题集、采样与 prompt style 身份，改变设置必须换目录。提示超长会明确报错，不静默截断/丢题。

## 5. 查看结果

### 已测得的提速与推荐设置

在这台四卡 L40S 上，用 LiveCodeBench 的同一组前 32 道题比较了两种 TP=4 配置。原 `batch-size=2 / max-num-seqs=8` 生成了 25342 个输出 token，用时约 67.58 秒，约 375 token/s；改成 `batch-size=8 / max-num-seqs=16` 后生成了 20507 个输出 token，用时约 19.59 秒，约 1047 token/s，生成吞吐约为原来的 2.79 倍。此口径排除模型加载和图捕获；该小样包括初始化的进程总耗时由约 141.85 秒降至 81.45 秒，约 1.74 倍。不是整项 benchmark 的保证加速比。

两次都保留完整题集定义、每题 4 次采样、temperature=1、top-p=1、seed=42、8192 token 回答上限及官方判题规则。调整 batch 会改变随机采样序列，输出总 token 数不同，不能把它描述为“完全相同回答的加速”。当前优先推荐上面的 TP=4 大批量命令；CPU 判题的 32 worker 是待观察的并发起点，没有把它计入上述生成加速比。

原完整测评已完成并保留在 `results/deepseek-coder-1.3b-base-pass4`。只想查看原分数无需重跑；新目录用于后续模型或独立吞吐对照。不要为了缩短运行时间默认缩短回答、减少题目、采样或测试用例。

### 可选的四卡分题模式

DeepSeek-Coder 1.3B 可在一张 L40S 上加载完整模型。增加 `--parallel-mode data` 会为每张可见 GPU 启动一个独立模型副本（每个进程 TP=1），按题目全局索引分成互不重叠的分片，最后合并到原格式。实测没有比 TP=4 大批量更快：32 题 DP4 生成了 28984 个输出 token，从各卡开始生成到最慢卡完成约 62.09 秒，聚合约 467 token/s，进程总耗时约 101.84 秒；TP=4 大批量在此小样中约 1047 token/s、总耗时约 81.45 秒。因此仍推荐 TP=4 大批量，DP4 保留为可选对照。任务量、初始化差异、回答长度与静态分片负载不均会影响结果，不将该结论推广到其他模型或所有数据集。保留 temperature、top-p、pass@4 的 4 次采样、8192 回答长度和全部测试用例。

服务器原始吞吐对照保存在 `results/performance-pilot-20261110-32/throughput.json`，只有生成结果，不参与正式 pass@4 分数。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/evaluate_code.py \
  --model-path models/deepseek-coder-1.3b-base \
  --benchmark all --prompt-style completion \
  --parallel-mode data \
  --samples 4 --temperature 1.0 --top-p 1.0 --seed 42 \
  --max-prompt-tokens 8192 --max-tokens 8192 \
  --batch-size 8 --max-num-seqs 16 --workers 32 \
  --output-dir results/deepseek-coder-1.3b-base-pass4-dp4
```

每个单卡 worker 一次向引擎提交 8 道题（共 32 个采样请求），引擎最多同时运行 16 条序列；32 是 CPU 判题并发，与 GPU 数不同。该选项用于后续对照，避免把 GPU 利用率百分数当作实际吞吐。

`generation_shards/workers_4/shard_XXX/` 保存各卡断点、`worker.log` 与吞吐数据；主进程及进度窗口显示各卡已保存题数之和。已有串行的完成前缀可以保留，所有分片覆盖完整且通过身份校验后才写入总 `generations.jsonl`。续跑分片时保持相同可见 GPU 数，不能同时运行多个进程写同一个输出目录。不要在本轮快结束时为切并行方式而重跑；推荐先跑完已有基线，再使用上面独立的新目录。

每个生成进程的 `generation.performance.json` 分开记录模型初始化和实际生成耗时、输出 token 数及 tokens/s，方便实测比较。采样次数、提示与计分规则不变，但 batch/并行布局改变后，固定 seed 不保证逐条答案与旧运行完全相同。

大模型只有在单卡能容纳权重、激活和 KV cache 时才能使用此模式；默认仍为 `--parallel-mode tensor`，兼容原来的多卡模型分片。将回答上限从 8192 减到 2048 或新增函数结束停止规则会改变测评协议，不作为默认提速选项。

新启动的生成进程显示整个 benchmark 的题目进度条（包含总题数、已完成题数、速度和 ETA），断点续跑从已有题数起步；隐藏 vLLM 每个小批次的 `8/8` 进度。LCB 判题也按整项题目数显示；HumanEval+/MBPP+ 保留官方 EvalPlus 以回答数计的判题进度。

正在运行的旧进程无需重启，可另开一个终端，在同一项目和环境下执行：

```bash
python scripts/watch_code_progress.py \
  --output-dir results/deepseek-coder-1.3b-base-pass4
```

该窗口显示三项数据集及全部题目的生成进度、已保存的判题进度。只读取新增日志，不使用 GPU，不修改回答；在此窗口按 Ctrl+C 只关闭进度显示。EvalPlus 的实时判题条仍在原运行窗口，最终保存的题数会在本窗口更新。

```bash
python -m json.tool results/deepseek-coder-1.3b-base-pass4/summary.json
```

每项的 `pass@4` 为 0–1 的比例，乘以 100 是百分数。子目录分别保留回答、判题详情和 summary。中断后用同一命令续跑；只重判时可去掉 `--model-path` 并添加 `--grade-only`，其余 benchmark、prompt style 和采样参数保持一致。

本地和服务器的生成、断点合并、设备隔离、进度及判题协议共 39 项检查通过；完整 GPU 测评已在服务器完成。吞吐小样的范围与限制见上文，后续模型的实际分数和速度仍以其独立运行结果为准。
