# opd-icml

DeepSeek-Coder 1.3B / 6.7B / 33B Base → Instruct、Qwen3-1.7B / 4B、Qwen3-8B Base → 32B 教师，以及 Gemma-3-12B PT → IT 的 on-policy distillation 实验代码。所有命令以此目录为项目根目录。训练采用本目录随附的修改版 verl。绘图输入与论文源码可以单独使用。

四张 H200 服务器上的环境安装和七组训练加评测，直接按下面的 [H200 操作指南](#h200-操作指南) 执行。后文的 A100/L40 示例及论文历史 GPU 数用于对应原实验配置。

| 目录或入口 | 内容 |
|---|---|
| `train.py`、`configs/<scale>/` | 七组模型的训练入口、Hydra 配置，以及学生和教师模型的固定 revision |
| `runtime/verl/workers/actor/` | Top-16 OPD、token 诊断、表征谱、梯度一致性和表征稳定性停止判据 |
| `runtime/verl/trainer/ppo/` | rollout、优化更新、复用、验证集与 checkpoint 管理 |
| `scripts/` | 安装 GPU 环境、准备数据、H200 训练加评测入口、数学 Mean@8 和代码 pass@4 |
| `analysis/`、`data/` | 结果提取、作图、参数层附录，以及绘图输入和六项 benchmark 题目 |
| `paper/main.tex` | 论文入口及当前正文和附录引用的文件 |

## H200 操作指南

**四卡整套实验推荐入口：** `bash scripts/h200/run_all.sh --devices 0,1,2,3 --output runs/h200-suite/run1`。先加 `--dry-run` 可以查看资源分配，不下载或启动训练。四个单卡实验在 CPU/主存预算允许时并行，随后 Qwen 8B、Gemma 12B、DeepSeek 33B 分别独占四卡；代码生成完成后交给单独 CPU 队列判题，与后续 GPU 实验重叠。只跑部分模型可加 `--scales 1.7b 4b 8b`。完整参数和资源说明见 [H200 调度说明](scripts/h200/README.md)。

H200 入口默认采用按模型设置的 `throughput` 起始预设；可用 `--profile conservative` 恢复原来 micro-batch=1、生成并发16及卸载策略。它们保持106轮、batch128、学习率、loss、模型精度、诊断样本数与频率。吞吐预设尚无真实 H200 性能/显存测量，不声称是最优值；改变批处理和生成调度可能改变浮点规约与采样轨迹。

本节对应**单机四张 H200**，按学生模型大小固定卡数：**≥8B 用四卡，其余用单卡，训练和评测生成保持相同卡数及 GPU 列表**。四卡 rollout/评测为 TP=4，单卡为 TP=1；代码测试用例判题在 CPU 上执行。

### 1. 进入项目并创建环境

以下命令在 Linux x86_64 服务器执行。先进入包含 `train.py`、`configs/`、`runtime/`、`scripts/` 的目录；若上传时保留了双层目录，它通常是 `opd-icml/opd-icml/`。每次重连服务器后，都先回到这个目录并激活环境。

先确认能看到四张卡，以及主存和磁盘空间：

```bash
nvidia-smi
nvidia-smi topo -m
free -h
df -h .
```

创建环境时，下面两种方式任选一种。服务器已有 Conda 时：

```bash
conda create -n opd-h200 python=3.10 -y
conda activate opd-h200
```

如果不用 Conda，但已有 `python3.10` 和 venv 支持：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

后续安装、训练、合并和评测都使用这个环境的 `python`。本流程固定 Python 3.10、PyTorch 2.6.0+cu124、vLLM 0.8.5.post1、Transformers 4.52.4，FlashAttention/FlashInfer 使用与之匹配的预编译 wheel。

### 2. 安装训练和评测依赖

在已激活的环境、项目根目录下，依次执行：

```bash
bash scripts/setup_code_eval.sh
bash scripts/setup_opd_training.sh
bash scripts/setup_code_benchmarks.sh
```

第一步安装固定版本的 PyTorch/vLLM 和基础评测依赖；第二步在同一环境补齐训练、数学判题、注意力内核，并以 editable 方式安装本项目的 `runtime/`；第三步检查代码 benchmark 使用的 EvalPlus 版本。安装需要访问 PyPI、PyTorch wheel 源和 GitHub。这里无需再运行会另建 `.venv` 的 `scripts/setup.sh`。

检查环境：

```bash
python -m pip check
python - <<'PY'
import sys
import torch
import transformers
import vllm
import ray
import flash_attn
import flashinfer

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Transformers:", transformers.__version__, "vLLM:", vllm.__version__)
print("Visible GPUs:", torch.cuda.device_count())
assert sys.version_info[:2] == (3, 10)
assert torch.__version__ == "2.6.0+cu124"
assert transformers.__version__ == "4.52.4"
assert vllm.__version__ == "0.8.5.post1"
assert torch.cuda.is_available()
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

如果安装或导入失败，先检查当前 `python` 是否来自刚创建的环境，以及 GPU/驱动是否对该环境可见。不要单独升级 Torch、vLLM 或注意力内核；安装脚本中的 wheel、CUDA runtime 和 ABI 是配套的。H200 启动脚本本身不会安装依赖。

### 3. 准备模型权限、缓存和数据

模型按 `configs/<scale>/models.json` 的固定 revision 下载并复用 Hugging Face 缓存。可选：如果模型缓存需要放到大容量持久化磁盘，在登录和下载前设置 `HF_HOME`，并在后续终端保持相同设置。下面路径需替换成服务器实际挂载点：

```bash
# export HF_HOME=/data/huggingface
```

Gemma 需要先在对应模型页面取得访问权限，再在服务器交互登录。若设置了 `HF_HOME`，先设置再登录，以保持缓存和登录凭据的位置一致：

```bash
huggingface-cli login
```

完整启动脚本会自动准备默认训练数据、下载模型，并准备缺失的代码 benchmark。也可以先单独准备，便于在正式占用训练卡之前完成下载和检查：

```bash
# 数学训练集；数学评测题 data/benchmarks.jsonl 已随仓库提供。
python scripts/prepare_dapo14k.py --output data/dapo14k_train.parquet

# DeepSeek 默认使用的代码训练/验证集，以及三项代码评测数据。
python scripts/prepare_eurus_code.py
python scripts/prepare_code_benchmarks.py

# 按需提前准备某组学生和教师；这些命令不启动训练。
python scripts/prepare_models.py --scale deepseek-coder-1.3b
python scripts/prepare_models.py --scale deepseek-coder-6.7b
python scripts/prepare_models.py --scale deepseek-coder-33b
python scripts/prepare_models.py --scale 1.7b
python scripts/prepare_models.py --scale 4b
python scripts/prepare_models.py --scale 8b
python scripts/prepare_models.py --scale gemma3-12b
```

只准备本次要运行的任务和模型即可。已有数据、模型和 benchmark 会按各准备入口的校验规则复用。DeepSeek 会准备使用教师模板及双 EOS 的学生运行目录；Gemma 会提取文本权重，额外占用一份文本模型的磁盘空间。33B 模型下载、FP32 训练 checkpoint、合并权重及诊断缓存都需要预留主存和磁盘；这些资源不能只按 BF16 推理权重大小估算。

### 4. 检查并运行七组实验

先以 Qwen 8B 为例查看配置，确认输出中的 `training_gpus`、`evaluation_gpus`、`rollout_tp`、`evaluation_tp` 都为 4，再正式运行：

```bash
bash scripts/h200/train_eval_qwen3_8b.sh --dry-run
bash scripts/h200/train_eval_qwen3_8b.sh
```

`--dry-run` 不下载、不创建运行目录、不使用 GPU。正式运行按顺序完成：准备数据 → 准备模型 → 训练 → 检查最终 checkpoint 步数和所有 rank 分片 → 合并 HF 权重 → 评测。训练失败时不会继续合并或评测。

七个脚本如下。按需选择；若执行整个代码块，会依次串行运行七组，每组有独立默认输出目录：

```bash
# 单卡：DeepSeek-Coder 1.3B Base → 1.3B Instruct；代码 pass@4。
bash scripts/h200/train_eval_deepseek_coder_1_3b.sh

# 单卡：DeepSeek-Coder 6.7B Base → 6.7B Instruct；代码 pass@4。
bash scripts/h200/train_eval_deepseek_coder_6_7b.sh

# 四卡：DeepSeek-Coder 33B Base → 33B Instruct；代码 pass@4。
bash scripts/h200/train_eval_deepseek_coder_33b.sh

# 单卡：Qwen3 1.7B Base → 1.7B；数学 Mean@8。
bash scripts/h200/train_eval_qwen3_1_7b.sh

# 单卡：Qwen3 4B Base → 4B；数学 Mean@8。
bash scripts/h200/train_eval_qwen3_4b.sh

# 四卡：Qwen3 8B Base → 32B 教师；数学 Mean@8。
bash scripts/h200/train_eval_qwen3_8b.sh

# 四卡：Gemma-3 12B PT → 12B IT；数学 Mean@8。
bash scripts/h200/train_eval_gemma3_12b.sh
```

DeepSeek 默认评测 LiveCodeBench、MBPP+、HumanEval+，每题采样 4 个回答；Qwen/Gemma 默认评测 `data/benchmarks.jsonl`，每题采样 8 个回答。可加 `--task code` 或 `--task math` 切换任务，卡数仍由学生规模固定。Qwen 后训练教师的官方仓库名不带 `Instruct` 后缀，具体配对见各组 `models.json`。

单卡默认选 GPU 0，四卡默认选 0,1,2,3。已有 `CUDA_VISIBLE_DEVICES` 时按其顺序取所需卡数；卡数不足会报错。用 `--devices` 明确指定时，必须正好提供该组所需的 1 张或 4 张卡：

```bash
# 指定 GPU 2，训练和评测都只使用这一张。
bash scripts/h200/train_eval_qwen3_4b.sh --devices 2

# 训练和评测都使用四张卡。
bash scripts/h200/train_eval_qwen3_8b.sh --devices 0,1,2,3
```

四卡实验应独占四张卡。并行运行单卡实验时，为各实验分配不同的 `--devices` 和输出目录，同时注意共享主存占用。

### 5. 常用参数和本地资源

七组脚本默认都是 `paired-full`：全局 batch/mini-batch=128、106 轮、每轮一次更新、学习率 1e-5。默认 `throughput` 下，1.3B/1.7B/4B 的学生与教师 micro-batch=4、rollout 并发64，并关闭 optimizer/teacher 阶段卸载；6.7B/8B/12B 为2和32，保留卸载；33B为1和16，保留卸载。梯度检查点和 rollout 显存比例0.30保留。逐 token 参数梯度默认每次抽样256个token（64 high-KL + 192 remaining），保留 hidden 统计。

下面示例使用独立输出目录，并显式切换为全量逐 token 参数梯度：

```bash
bash scripts/h200/train_eval_qwen3_8b.sh \
  --devices 0,1,2,3 \
  --parameter-token-gradients exact-all \
  --output runs/h200/qwen8-math-exact
```

使用已有本地权重和训练数据时，例如：

```bash
bash scripts/h200/train_eval_deepseek_coder_6_7b.sh \
  --student /data/models/deepseek-coder-6.7b-base \
  --teacher /data/models/deepseek-coder-6.7b-instruct \
  --model-dir /data/models/opd-prepared \
  --data /data/eurus/eurus_code_train.parquet \
  --val-data /data/eurus/eurus_code_validation.parquet \
  --code-benchmarks /data/code_benchmarks \
  --output runs/h200/deepseek6.7-code-local
```

`/data/...` 示例路径需要替换成已有资源位置。`--model-dir` 存放 DeepSeek 学生运行目录或 Gemma 文本权重；仅缓存下载可用 `--cache-dir`。自定义数学评测题用 `--math-benchmarks`；自定义代码 benchmark 目录用 `--code-benchmarks`，其中应包含原准备流程生成的三项题目和 manifest。

可调参数还包括 `--micro-batch-size`、`--teacher-micro-batch-size`、`--optimizer-offload on|off`、`--teacher-offload on|off`、`--rollout-max-num-seqs`、`--rollout-gpu-memory-utilization`、`--eval-max-num-seqs`、`--eval-batch-size`、`--code-workers` 和 `--score-workers`；`--eval-batch-size` 只用于代码评测。资源控制支持 `--ray-num-cpus`、`--ray-object-store-gib`、`--gradient-cache-gib`、`--cpu-ids`。这些选项不会更改固定的单卡/四卡规则。完整选项可执行：

```bash
python scripts/run_h200_experiment.py --help
```

### 6. 单独训练、重试评测和查看结果

默认 `--stage all` 会训练后自动评测。也可分两次执行；第二次使用相同输出目录、任务和 GPU 设置：

Code 的 `all`/`eval` 先生成完三项 benchmark，再统一判题。`--stage train-generate` 完成训练、合并和全部生成后退出，`--stage generate` 只对已有最终 checkpoint 合并及生成；`--stage grade` 只运行 CPU 判题，不加载模型或检查 GPU。总调度使用这些阶段，在保留 pass@4 题集、采样数、长度和判题标准的同时释放 GPU。Math 继续使用现有 Mean@8 生成与评分流程。

```bash
bash scripts/h200/train_eval_qwen3_8b.sh \
  --stage train --devices 0,1,2,3 --output runs/h200/qwen8-math-run1

bash scripts/h200/train_eval_qwen3_8b.sh \
  --stage eval --devices 0,1,2,3 --output runs/h200/qwen8-math-run1
```

`--stage eval` 只读取该组的最终训练 checkpoint，不重新训练；合并完成且标记匹配时复用 `merged/`，评测沿用原入口的回答缓存和判题进度。评测中断后可以重试第二条命令。若合并中断，留下的 `merged/` 没有完成标记，需先将它移到其他位置，再运行 `--stage eval`。训练中断后当前入口不支持续训已有输出目录；重新训练请指定新的 `--output`。

默认输出根目录是 `runs/h200/<scale>/<task>-paired-full/`，传入 `--output` 可替换这个根目录：

```text
<output>/
  training/checkpoints/       最终训练 checkpoint 与 latest 步数标记
  training/training_records/  训练回答和监测记录
  training/replication/       token、参数梯度及表征诊断
  merged/                    合并后的 HF 学生模型
  eval/final_mean8.jsonl      数学回答与评分
  eval/final_mean8.summary.json
  eval/code_pass4/summary.json
  logs/                      数据准备、训练、合并和评测日志
```

数学和代码输出只会出现本次选择任务对应的评测文件。例如查看默认 Qwen 8B 数学训练日志和最终汇总：

```bash
tail -F runs/h200/8b/math-paired-full/logs/train.log
# 在另一终端，或退出日志查看后读取已完成的汇总。
cat runs/h200/8b/math-paired-full/eval/final_mean8.summary.json
```

6.7B 单卡和 33B 四卡仍需在真实 H200 上验证完整训练更新及梯度诊断的显存峰值。当前配置是保守起点，本地 CPU/dry-run 检查不代表 H200 已实跑通过；脚本不会在显存不足时自动增加卡数或改变实验预算。

## 训练配置

| `--experiment` | 论文用途 | 新 rollout 轮数 | 每轮更新 | 原运行 GPU 数（1.7B / 4B） |
|---|---|---:|---:|---:|
| `paired-full` | RQ1–RQ3。配对研究中的 FULL，使用 128 道排除训练的验证题 | 106 | 1 | 2 / 2 |
| `original-full` | RQ4 耗时与 benchmark 的 FULL 基线 | 106 | 1 | 1 / 2 |
| `early-stop` | RQ4 Adaptive early stop，每五步检查一次稳定性 | 上限 106，原运行在 40 / 50 步停止 | 1 | 1 / 2 |
| `reuse` | RQ4 Rollout reuse，按相同优化更新预算比较 | 53 | 2 | 1 / 2 |

原有 1.7B / 4B 配置来自成功运行的训练日志中实际解析的参数。paired FULL 的 rollout batch 是 128；原始研究的 batch 是 160，使用其原有的 128/32 协议。两套协议分别有自己的 FULL 参考结果。模型和数据路径在运行时注入，日志默认输出到控制台。启动入口用于新实验，每个配置使用独立输出目录。

## RQ1 全量逐 token 参数梯度

针对 4×L40、主存 512 GB 或以上，可在下面的训练命令中增加 `--hardware l40x4`，使用后文的四卡启动预设。

四组模型的 `paired-full`（math 和 code）默认开启 `gradient_study.parameter_tokens.enabled=true`，模式为 `exact_all`。测量时机与 hidden 统计一致：第 1 步、每 5 次更新、最后一步。每次覆盖 hidden 统计中的全部有效 response token，不抽样、不做随机投影估计。其他实验配置保持原设置；有 `gradient_study` 的配置也可通过下面的命令显式开启。

对每个 token 单独反传其未平均、未裁剪的条件 Top16 KL，计算 `parameter_gradient_energy = sum_theta (d loss_i / d theta)^2`。范围是所有参与训练的 backbone、embedding、归一化和输出头参数；共享参数只计一次。FSDP 下所有 rank 同步计算同一个 token，再累加不重叠梯度分片的平方范数。梯度平方和用 FP64 累加；“精确”指直接逐 token 自动微分，不使用范数估计器，仍采用训练环境本身的数值精度。

逐 token 测量在启用 `replication17.compact_head` 时，仅对目标预测位置的 hidden 和该位置缓存的 16 个词表 ID 做输出投影，再计算该 token 的 KL；不再投影其他 response 位置或构造它们的词表并集。`exact_all` 和 `kl_stratified` 共用此路径，保留所需的 Transformer 上下文、完整参数反传和模型自身的 logits 后处理；正常训练仍计算全部训练位置的损失。

同一回答内的待测 token 默认每 4 个复用一次前向计算图（`parameter_tokens.graph_group_size=4`），前缀保留到该组最后一个目标位置。每个 token 仍分别反传、计算完整参数梯度平方范数并清空梯度，最后一个 token 反传后释放计算图；不减少待测 token 数，不把多个梯度相加后求范数。每个位置只投影自己的 16 个词表 ID。可按显存和实测耗时调整 `--parameter-token-group-size`，设为 `1` 恢复逐 token 前向；这不是自动显存预算或 OOM 重试机制。

参数统计与 hidden 统计写在同一个 `replication/token_audit/step_XXX_inner_N.json` 和 `.npz` 中。每个现有 token 分组（包括 KL top 1/5/10/20%、固定 step-1 阈值、位置等）增加 `quantities.parameter_gradient_energy`，含总和、均值和能量占比；还保存参数梯度 RMS、Frobenius 范数、平均 L2 范数、能量集中度，以及逐 token 标量和 row/pos。高/中/低 KL 分组也增加参数能量和 RMS。`parameter_gradient_measurement` 记录全量覆盖、额外反传次数、耗时和定义；不保存整个参数梯度向量，不额外调用 teacher，不执行诊断 optimizer step。

报告中的 Top-1/5/10/20% 分组对每个分数只做一次稳定降序排序；KL 高/中/低分组保留原来的稳定升序排序和并列处理。完整报告与抽样报告在主 rank 上共享当前更新的分组和 KL 排序结果。teacher 熵复用输出统计中已算出的双精度 log-probability，不重复计算 `log_softmax`。这些复用不跨参数更新，保持原有分组成员、报告字段和统计精度。

跨 batch／同题多回答的参数梯度比较按主存预算缓存当前调用内的部分精确 FP32 梯度快照，优先保留复用次数更多的梯度，用完即释放。四个子集分别保留 0、1、2、3 份可复用梯度时，求导次数为 10、7、5、4；所有 rank 使用共同允许的最小缓存量。仍使用原来的逐对内积顺序、参数分组和精度，缓存不跨 checkpoint、数据集或归一化方式复用。预算按所有 rank 合计不超过可用主存的四分之一及 64 GiB 中的较小值，包含缓存、工作向量及一份捕获临时空间；同时检查可读的 cgroup 内存限制。按 40 亿参数估算，默认上限允许缓存两份梯度，但仍需足够的可用主存；内存不足或无法读取内存信息时自动减少缓存或恢复重算。可用 `OPD_GRADIENT_CACHE_MAX_GIB` 调整上限，设为 `0` 恢复原路径。该复用不改变 token 抽样模式、观测时点、loss、梯度精度或 optimizer 状态。

KL 分组的完整样本与等样本数分析在选中的行、顺序和归一化完全相同时，复用同一份谱计算结果。报告字段及数值定义保持不变。以上改动仅减少重复计算，未修改计算图复用、FSDP 反传钩子或分布式梯度归约。

**代价仍大：测量 N 个 token 仍需 N 次全参数反传，显式前向次数降为各回答 `ceil(待测 token 数 / graph_group_size)` 之和。** 梯度检查点每次反传仍可能重算内部前向，不能按前向次数比例推断总提速；分组还会延长计算图的存活时间，并让组内较早 token 使用较长前缀。所有诊断在正式训练 backward 前运行，并恢复 RNG、模块模式和清空诊断梯度；正式训练仍使用原损失。控制台及 `parameter_gradient_measurement` 记录 `forward_calls`、有效 `graph_group_size` 和 `graph_reuse_fallback`，前向计数不含检查点重算。原有 `actor/grad_norm` 是 batch 参数梯度范数；逐 token 平方范数之和不等于 batch 梯度平方范数，也不代表 AdamW 更新能量。

```bash
python train.py --scale 1.7b --experiment paired-full --parameter-token-gradients exact-all
python train.py --scale 4b --task code --experiment paired-full --parameter-token-gradients exact-all
# 显存紧张或回归对照：保留完整逐 token 测量，关闭计算图复用
python train.py --scale 4b --task code --experiment paired-full --parameter-token-group-size 1
# 快速调试时显式关闭；RQ1 正式全量记录使用 exact-all
python train.py --scale 1.7b --experiment paired-full --parameter-token-gradients off
python analysis/extract_signal.py --run-17 runs/1.7b/paired-full --run-4 runs/4b/paired-full --output outputs/rq1.csv
python analysis/plot_rq1_rq4.py --rq1 outputs/rq1.csv
```

新 CSV 增加 `top_kl_decile_parameter_energy_pct`、参数梯度范数、hidden/参数 RMS 与 Frobenius 范数和诊断开销；有新数据时额外生成 `rq1_parameter_energy.pdf` 和 `rq1_gradient_norm.pdf`。旧日志缺失的新字段留空，不回填或编造历史参数梯度数据。此统计不包含全程移除高 KL 监督的消融训练。

当前模型配对使用零 dropout，记录器会检查这一点，保证确定性诊断与训练的目标一致。当前支持项目的 FSDP1 分片方案（`use_orig_params=false`），尚不支持 FSDP2/DTensor 或混合分片。服务器可先在两张 GPU 上运行 `python tests/check_parameter_tokens_distributed.py --fsdp` 验证分片范数；无 GPU 时去掉 `--fsdp` 可验证两进程 CPU DDP。

计算图复用的 FSDP1 适配依赖 PyTorch 2.6/2.7 的内部反传钩子生命周期，其他版本、编译模型、参数 CPU offload 或项目的 activation offload 会自动回退到组大小 1 并记录原因。已在本地 PyTorch 2.7.1 CPU 上验证按层 FSDP 分片、共享 embedding、非重入梯度检查点及 BF16 混合精度；项目固定的 PyTorch 2.6 CUDA 配置仍需在服务器运行 `python tests/check_parameter_graph_reuse_distributed.py --cuda --fsdp --checkpoint --mixed-precision` 验证。CPU 可运行同一命令并去掉 `--cuda`，Windows 下需先设 `USE_LIBUV=0`。本地测试不代表 GPU 吞吐或显存实测。

## 安装和准备数据

### 4×L40 启动预设

`--hardware l40x4` 在启动时显式覆盖四卡 FSDP、rollout TP=4、actor/ref/log-prob micro-batch=1、梯度检查点、AdamW optimizer CPU offload、teacher 参数 CPU offload，并关闭 teacher 编译、将 rollout 并发限制为 16。保持全局 rollout batch、训练目标、长度预算及诊断开关；不改写记录的原配置文件。四组模型的 math/code `paired-full` 均可用，全量精确参数梯度统计仍默认开启。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 先查看四卡与全量梯度开关，不加载模型
python train.py --scale 4b --experiment paired-full --hardware l40x4 --dry-run
# 正式 math / code：按顺序分别运行，每个实验使用整个四卡节点
python train.py --scale 4b --experiment paired-full --hardware l40x4 --output runs/l40x4/4b/math
python train.py --scale 4b --task code --experiment paired-full --hardware l40x4 --output runs/l40x4/4b/code
```

可把 `4b` 换成 `1.7b`、`8b` 或 `gemma3-12b`。512 GB 或以上主存有利于卸载及梯度诊断缓冲，但这仍是待服务器验证的启动配置，不保证 8B/12B 的实际显存峰值或吞吐。L40 每卡 48 GB、无 NVLink，不能将四卡直接当成单卡 192 GB 使用；完整逐 token FSDP 反传的通信和计算成本都很高。先观察首次测量的 token 数、进度和耗时，再估算完整实验时间。仅为快速检查普通训练时，可显式加 `--parameter-token-gradients off`；正式全量记录保留默认开启。

保留现有 CUDA 12.4 / PyTorch 2.6 / vLLM 0.8.5.post1 环境。运行前可执行 `nvidia-smi` 和 `nvidia-smi topo -m` 查看驱动、四张卡和 PCIe 拓扑；不要仅因没有 NVLink 就固定设置 `NCCL_P2P_DISABLE=1`。包含单卡 disposable shadow probes 的 `1.7b/original-full` 不支持该四卡预设，入口会明确报错，不会自动删除探针。

在 Linux CUDA 训练环境中，从本目录执行。安装脚本使用 Python 3.10、PyTorch 2.6.0 / cu124、vLLM 0.8.5.post1 和原有项目的依赖版本。

```bash
bash scripts/setup.sh
source .venv/bin/activate
python scripts/prepare_dapo14k.py --output data/dapo14k_train.parquet
```

默认使用模型配置中记录的 Hugging Face revision。已有模型可通过 `--student /path/to/student --teacher /path/to/teacher` 指定。训练数据可通过 `--data` 指定。首次下载模型和训练集需要网络。

## 新增模型配对与模型准备

`--scale 8b` 对应学生 `Qwen/Qwen3-8B-Base`、教师 `Qwen/Qwen3-32B`（该系列后训练模型的官方仓库名不带 `Instruct` 后缀）。`--scale gemma3-12b` 对应学生 `google/gemma-3-12b-pt`、教师 `google/gemma-3-12b-it`。两组的具体 commit 固定在各自 `configs/<scale>/models.json`。

先查看模型配置，或单独准备模型，不需要训练 parquet，也不会启动训练：

```bash
python train.py --scale 8b --prepare-only --dry-run
python train.py --scale 8b --prepare-only
python train.py --scale gemma3-12b --prepare-only
```

也可使用 `python scripts/prepare_models.py --scale 8b`；两个入口都支持 `--student`、`--teacher` 本地目录，`--cache-dir` 自定义 Hugging Face 缓存，及 `--model-dir` 自定义 Gemma 文本权重输出根目录。未提供本地目录的模型会按固定 revision 下载，已缓存的文件会复用。Gemma 需要先在对应 Hugging Face 模型页面获得访问权限，再在服务器运行 `huggingface-cli login`，或提供已获得的本地模型目录；命令和配置中不保存 token。

Gemma 官方 12B checkpoint 包含视觉模块，本项目的数学任务使用纯文本。准备阶段按 shard 提取语言模型权重到 `models/gemma3-12b/{student,teacher}/`，保留原始 tensor 数值与 dtype，供 Transformers、vLLM、checkpoint 合并和现有数学评测共用。输出采用 `Gemma3ForCausalLM`，统一使用 Gemma 的 user/model turn 格式和 `<end_of_turn>` EOS。转换会额外占用一份文本权重的磁盘空间，按一个原始 shard 的大小使用 CPU 内存；重复运行会复用同源的完成输出。`models/` 已加入 `.gitignore`。

数学与代码的 chat 评测按训练规则套用模板，再以 `add_special_tokens=False` 编码并将 `prompt_token_ids` 传给 vLLM，避免 Gemma 模板 BOS 被字符串路径再次添加；模板轮次标记和模型 EOS 配置保留。新生成记录使用 `chat_template_token_ids_v1` 编码标识，旧 chat 缓存不能混入新生成任务；重新生成请使用新的输出路径，历史回答仍可通过数学 `--regrade_only` 或代码 `--grade-only` 纯评分。Base/completion 仍由 tokenizer 添加原有 BOS 后传 token IDs。

新增两组各提供 `paired-full`、`original-full`、`early-stop`、`reuse` 四种配置。例如：

```bash
python train.py --scale 8b --experiment paired-full --dry-run
python train.py --scale 8b --experiment paired-full
python train.py --scale gemma3-12b --experiment paired-full
```

新配置默认使用单节点 4 GPU、rollout TP=4、每 GPU micro-batch=1、梯度检查点、actor optimizer CPU offload 和 teacher parameter CPU offload。GPU 数需与配置中的 `trainer.n_gpus_per_node`、`actor_rollout_ref.rollout.tensor_model_parallel_size`、`actor_rollout_ref.actor.policy_loss.replication17.distributed_world_size` 保持一致。它们是新实验起始配置；显存、主存和吞吐需在训练服务器实测，并非已有成功运行的资源记录。

新增配对的 `paired-full`、`early-stop`、`reuse` 使用 batch=128 和独立固定 128 道验证题；`reuse` 使用 53 轮、每轮 2 次更新，并在本次运行中采集自己的表征参考，无需先复制 FULL 子空间文件。`original-full` 保留 batch=160 的 128/32 协议。Gemma 学生和教师使用相同词表、模板和 EOS；Qwen 跨规模配对在准备阶段也会检查词表 token IDs 一致。

CPU 兼容性测试（使用随机微型模型，无需下载真实权重）：

```bash
python -m pytest tests/test_model_assets.py -q
python tests/check_distributed_cpu.py
```

## 启动实验

四张 H200 上的七组训练加评测入口见 [scripts/h200/README.md](scripts/h200/README.md)：学生 ≥8B 固定四卡，其余固定单卡；同组训练、rollout 和评测生成使用相同 GPU 列表。每个脚本自动串联训练、最终 checkpoint 检查、HF 权重合并和评测，支持 `--dry-run`。

数学训练的专用脚本为 `scripts/train_math.sh`。先激活现有训练环境，可直接编辑脚本顶部的模型、实验、硬件、batch、学习率、生成并发和梯度诊断参数，也可用环境变量覆盖：

```bash
# 默认 Qwen 1.7B paired-full、单卡 A100；先查看配置，不加载模型。
bash scripts/train_math.sh --dry-run
bash scripts/train_math.sh

# 四卡模型配置示例：先检查，正式运行时去掉 --dry-run。
SCALE=4b HARDWARE=l40x4 bash scripts/train_math.sh --dry-run
SCALE=8b HARDWARE=l40x4 bash scripts/train_math.sh --dry-run
SCALE=gemma3-12b HARDWARE=l40x4 bash scripts/train_math.sh --dry-run
```

`ROLLOUT_BATCH_SIZE` 与 `MINI_BATCH_SIZE` 默认 `auto`，继承所选实验的 batch：paired-full 为 128，original-full 为 160，早停/reuse 按对应 YAML。当前 Top16 OPD 要求一次 actor 调用覆盖整批 rollout，mini-batch 与 rollout batch 相同；带 32 道 probe 的原协议仍保留。`MICRO_BATCH_SIZE=1` 是每张卡一次前向/反向的题数，`ROLLOUT_MAX_NUM_SEQS=16` 是生成并发。可通过 `STUDENT_PATH`、`TEACHER_PATH`、`DATA_PATH` 和 `OUTPUT_DIR` 指定本地路径，路径带空格也支持。

脚本中 paired-full 默认 `PARAMETER_TOKEN_GRADIENTS=sampled`、`PARAMETER_TOKEN_SAMPLES=256`（64 high-KL + 192 remaining）；全量 hidden 统计保留。其他实验默认继承原 YAML 的诊断开关，避免对未开启 gradient-study 的协议强行添加参数诊断。设 `PARAMETER_TOKEN_GRADIENTS=inherit` 可严格继承 YAML，或明确选 `exact-all` / `off`。`HARDWARE=recorded` 继承 YAML GPU 数；未设置 `CUDA_VISIBLE_DEVICES` 时自动匹配所选卡数，用户显式设置时保留。

可以显式传入 GPU 数。数学脚本的 `NUM_GPUS=auto` 默认继承硬件预设/原 YAML，Code 脚本默认 1；显式 `NUM_GPUS` 同时覆盖 FSDP 进程数、诊断 world size 和默认 rollout TP，不按卡数放大全局 batch。`ROLLOUT_TP_SIZE` 可指定卡数的更小约数，例如 4 卡、TP=2 会有两组生成 worker；该选项不是保证更快，需实测。

```bash
# 数学，两卡；全局 batch 仍为 128
NUM_GPUS=2 bash scripts/train_math.sh --dry-run

# 数学，四卡；每卡训练 micro-batch=2，生成 TP=2
NUM_GPUS=4 ROLLOUT_TP_SIZE=2 MICRO_BATCH_SIZE=2 \
  bash scripts/train_math.sh --dry-run

# Code，四卡；无需改脚本里的 a100x1 内存预设名称
NUM_GPUS=4 bash scripts/train_deepseek_code_a100.sh --dry-run
```

Python 入口对应 `--num-gpus N --rollout-tp-size T`。启动前检查全局 batch、micro-batch、固定验证/探针面板和 GPU 数的整除关系；准备模型后检查 attention/KV heads 与 TP 的兼容性，实跑还检查可见 GPU 数。显式卡数不自动改 batch、长度、学习率、更新预算或诊断频率。已有 `CUDA_VISIBLE_DEVICES` 时沿其顺序选前 N 张；不足会报错，不自动更换卡号。

参数梯度诊断的 FP64 范数分块规约改为设备内累加，每张设备仅同步一个标量；支持的 Llama/Qwen/Gemma 因果模型只重算目标 token 的前缀，省去未来后缀的计算，`causal_prefix=false` 可用于回归对照。抽样 ID、权重、实际反传次数和定义保持原协议，日志新增诊断的实际/原后缀前向 token 量。checkpoint 发布检查支持任意正整数 world size，并继续要求每一 rank 的文件齐全。CPU 等价性测试不等于真实多卡吞吐实测；更换卡数、micro-batch 或生成调度会改变浮点规约和随机数分配，不承诺逐位相同或完全相同的生成答案。

```bash
SCALE=4b HARDWARE=a100x1 MICRO_BATCH_SIZE=2 \
  OUTPUT_DIR=runs/math/4b/paired-full-a100-micro2 \
  bash scripts/train_math.sh --dry-run
```

硬件和 micro-batch 的可启动配置不代表各模型都已完成显存验证；增加 batch/并发后需实测。改变全局 batch 会改变总训练题数，须为该实验另选输出目录。脚本不安装依赖，也不准备数据；训练数据仍使用原 DAPO 数学准备流程。旧 Python 命令继续可用：

只查看配置、预算、模型 revision 和目标路径：

```bash
python train.py --scale 1.7b --experiment paired-full --dry-run
```

实际训练命令如下。运行哪一条，就启动哪一个实验。把 `1.7b` 换成 `4b` 即使用对应规模的配置。

```bash
python train.py --scale 1.7b --experiment paired-full
python train.py --scale 1.7b --experiment original-full
python train.py --scale 1.7b --experiment early-stop
python train.py --scale 1.7b --experiment reuse
```

默认输出为 `runs/<scale>/<experiment>/`；也可以传入 `--output /path/to/new-run`。其中 `checkpoints/` 保存权重，`training_records/` 保存训练记录，`replication/` 保存该协议开启的诊断。早停步数由当前运行的规则判定。

## Code 实验：Eurus-2-RL-Data Code

DeepSeek-Coder 1.3B Base → Instruct 的四卡 OPD 已接入为 `--scale deepseek-coder-1.3b --task code`，安装训练依赖、准备模型/Eurus、全量梯度记录及启动命令见 [DEEPSEEK_CODER_OPD.md](DEEPSEEK_CODER_OPD.md)。学生运行目录复用原 Base 权重，共用 Instruct 模板与 EOT，保留原始 Base 测评目录。

直接评测 DeepSeek-Coder 1.3B Base 的独立环境、模型下载、三个 benchmark 准备及完整命令见 [DEEPSEEK_CODER_EVAL.md](DEEPSEEK_CODER_EVAL.md)。Base 模型使用 `scripts/evaluate_code.py --prompt-style completion`；默认 chat 评测保持原协议。

通过 `--task code` 切换到 [PRIME-RL/Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data) 的 code 子集。固定 revision 为 `9776b13264b5aaa0b16495fcf086a0a8d86fd655`，仅提取 `ability == "code"` 的题：训练 25,276 题、官方 validation 1,024 题，来源为 APPS、CodeContests、TACO、Codeforces。原始两个 parquet 混合了 math/code；准备脚本按 parquet row-group 统计跳过纯 math 分块，再精确按 ability 过滤，不将 math 题混入 code 训练。

在训练环境中先准备数据：

```bash
python scripts/prepare_eurus_code.py
```

输出 `data/eurus_code_train.parquet`、`data/eurus_code_validation.parquet` 和各自 `.manifest.json`。源文件较大，完整准备需要下载 code 测试用例并占用相应磁盘；本仓库不包含下载后的训练数据。若已有原始文件，可指定 `--source-dir /data/Eurus-2-RL-Data`，其中应有原始 `train.parquet`、`validation.parquet`；`--output-dir` 可更换输出位置。重复运行会校验版本、题量和输出 SHA-256 后复用；不覆盖其他协议或未完成的数据文件。

转换保留原始题目、Python 代码输出要求，以及完整的 `reward_model.ground_truth` JSON 测试用例（含可选 `fn_name`）。原始 PRIME system 指令折叠到第一条 user 消息，保留内容，统一兼容 Qwen 和 Gemma；不会添加数学 `\boxed{}` 指令。每题保存原始 split、原始行号、数据来源、prompt hash、测试用例数和固定 revision。

四组模型均支持四种 code 实验模式：

```bash
python train.py --task code --scale 4b --experiment paired-full --dry-run
python train.py --task code --scale 4b --experiment paired-full
python train.py --task code --scale 8b --experiment early-stop
python train.py --task code --scale gemma3-12b --experiment reuse
```

`--task` 默认仍是 `math`。Code 默认读上述独立训练/验证文件，并输出到 `runs/code/<scale>/<experiment>/`；可用 `--data`、`--val-data`、`--output` 指定路径。Code 配置为 `configs/<scale>/code-<experiment>.yaml`，继承已有模型/实验设置，默认 prompt 上限 2,048、response 上限 3,072；超长 prompt 会被过滤，实际参与训练的题数可能少于原始 25,276。新模型两组仍默认 4 GPU，1.7B / 4B 沿用其基础配置的 GPU 数。较长 code 序列的资源开销需在服务器实测。

蒸馏目标仍为学生 Top-16 支持上的 reverse KL；测试用例得分只用于监测，既不筛选训练 token，也不变成 PPO/GRPO 奖励。默认调用已有 PRIME Python 判题器，使用 Linux/Unix 训练环境，最多并发 8 个提交（`--code-workers` 可调整）。也可接已有 Sandbox Fusion 服务：

```bash
python train.py --task code --scale 4b --experiment paired-full \
  --sandbox-url http://127.0.0.1:8080/run_code --code-workers 8
```

监测指标为“一个回答通过该题全部测试用例”的比例；单个用例通过比例不会当作整题正确率。记录 `monitor/train_code_accuracy`、代码块格式率、编译/运行失败指标、token 长度、训练 KL、梯度与表征诊断；code 日志不使用数学 boxed accuracy 名称。`training_records/*.jsonl` 保存回答、得分、来源行号及测试集 hash/用例数，官方验证回答另存于 `validation_records/`；完整测试用例保留在数据 parquet 中，避免每轮重复写入。`paired-full` 及其 early-stop/reuse 保留独立固定 128 题诊断；另每 20 轮及结束时在官方 code validation 上做正确性监测。128 题诊断从训练池中排除，官方 validation 不参与训练。Code reuse 在本次运行中建立自己的表征参考，不依赖复制 FULL 子空间资产。

现有六项数学 benchmark 仍用于数学评测；新增 code 官方 validation 结果与数学均分分别记录。

CPU 数据和接入测试：

```bash
python -m pytest tests/test_code_task.py -q
```

## 三项 Code benchmark：pass@4

独立评测入口 `scripts/evaluate_code.py` 接入 LiveCodeBench、MBPP+、HumanEval+，主指标固定为 **pass@4**。默认每题独立采样 4 个回答（temperature=1、top_p=1、seed=42）：只要有一个回答通过该题完整测试，该题的 pass@4 即为 1。若指定 `--samples N` 且 N>4，则使用官方一致的无偏 pass@4 估计器；不把四次回答的平均正确率称作 pass@4。

- HumanEval+：164 题；Hugging Face `evalplus/humanevalplus` 固定 revision `d32357cf319e50e9c8d8dab5ea876c72b0fd321b`，官方测试发布版 `HumanEvalPlus v0.1.10`。
- MBPP+：378 题；Hugging Face `evalplus/mbppplus` 固定 revision `b2d74c91837c3f2a20c1299ae98133cbe7cfa077`，官方测试发布版 `MbppPlus v0.2.0`。题集 ID 与 HF 快照逐项核对，提示和测试采用官方修订版本。
- LiveCodeBench：`livecodebench/code_generation_lite` 固定 revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`，默认 `release_v6`，可指定 `release_v1` 至 `release_v6` 及起止日期。日期范围在结果 manifest 中保留，起止日期均包含当天。

在已激活的 Linux 训练/评测环境中安装判题依赖并准备数据：

```bash
bash scripts/setup_code_benchmarks.sh
python scripts/prepare_code_benchmarks.py
```

安装入口固定使用 `evalplus==0.3.1`。LiveCodeBench 的官方评测核心以固定 commit 随仓库提供，原文件校验值和许可证位于 `third_party/livecodebench/`。所有数据源、发布版本、下载校验值集中在 `configs/code_benchmarks.json`。

默认准备全部三项。LiveCodeBench v6 的六个原始 JSONL 文件合计约 4.49 GB，准备时还需要保存转换后的测试数据；首次下载需预留磁盘和时间。可分项准备：

```bash
python scripts/prepare_code_benchmarks.py --benchmark humanevalplus
python scripts/prepare_code_benchmarks.py --benchmark mbppplus
python scripts/prepare_code_benchmarks.py --benchmark livecodebench --lcb-release release_v6
```

若只比较某个时间窗口，可在 LCB 准备阶段指定，例如 `--start-date 2025-01-01 --end-date 2025-04-30 --output-dir data/code_benchmarks_2025`，评测时对应传入 `--data-dir data/code_benchmarks_2025`。`--dry-run` 只显示数据版本，不下载。数据默认放在 `data/code_benchmarks/`，下载缓存默认放在 `data/code_benchmarks_sources/`，均不提交到 Git；`--source-dir` 支持按原文件名/layout 提供离线数据源。不同版本或时间窗口请使用不同目录，避免混用。

将训练 checkpoint 按下文方法合并为 HF 模型后，运行三项 pass@4：

```bash
RUN="runs/code/8b/paired-full"
python scripts/evaluate_code.py --model-path "$RUN/merged" \
  --output-dir "$RUN/eval/code_pass4"
```

`--benchmark` 可单独选 `livecodebench`、`mbppplus`、`humanevalplus`；默认 `all`。也可将 `--model-path` 指向准备好的 Base/PT/IT 模型进行基线评测。默认 prompt 上限 4,096、回答上限 8,192；若某题超长，入口明确报错，不截断或丢弃该题，可按模型上下文和显存调整 `--max-prompt-tokens` / `--max-tokens`。

生成与判题在不同子进程执行：先保存模型回答，退出 GPU 生成进程，再启动 CPU 判题进程。可分别执行：

```bash
python scripts/evaluate_code.py --model-path "$RUN/merged" \
  --output-dir "$RUN/eval/code_pass4" --generate-only
python scripts/evaluate_code.py --output-dir "$RUN/eval/code_pass4" --grade-only
```

重判不需要模型目录或 GPU；需使用相同的题集、采样参数和 benchmark 选择。回答按任务保存，生成支持批次续跑；判题完成标记包含回答、测试和输出文件校验值。改变模型、题集或采样参数时使用新的输出目录。

HumanEval+/MBPP+ 通过固定版本 EvalPlus 的判题入口执行，单个回答必须同时通过 base 和 plus 两部分测试。EvalPlus 0.3.1 的控制台可能同时打印其自带的 pass@1；本项目从完整判题结果另行计算 pass@4。LiveCodeBench 使用官方 runner 检查全部公开和私有用例，按任务限制并发加载测试数据；私有测试不进入生成 prompt。测试用例、参考答案与模型生成输入分开保存。

每项输出 `generations.jsonl`、官方判题记录、`grades.jsonl`、`summary.json`；总览为 `<output-dir>/summary.json`，三个 benchmark 各自报告 pass@4（0 到 1）。保留题集 revision、LCB release/日期窗口、模型文件校验值、采样设置、回答长度和结束原因。数学 Mean@8 与 Code pass@4 分别报告。

接入与指标验证：

```bash
python -m pytest tests/test_code_benchmarks.py -q
```

## 合并模型与评测

在激活训练环境后，下面的命令评测 1.7B Rollout reuse。修改 `RUN` 可评测其他实验。读取 checkpoint 记录中的末步编号，避免混用 rollout 轮数和优化器更新数。

```bash
RUN="runs/1.7b/reuse"
STEP="$(cat "$RUN/checkpoints/latest_checkpointed_iteration.txt")"
python -m verl.model_merger merge --backend fsdp \
  --local_dir "$RUN/checkpoints/global_step_${STEP}/actor" --target_dir "$RUN/merged"
python scripts/evaluate.py --model_path "$RUN/merged" --input_file data/benchmarks.jsonl \
  --output_file "$RUN/eval/final_mean8.jsonl" --max_prompt_tokens 1024 --max_tokens 8192 \
  --temperature 1 --top_p 1 --n 8 --seed 42 --max_num_seqs 64 \
  --compact_output --scoring_mode lenient_answers_v1
```

`benchmarks.jsonl` 包含 MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23，共 905 道题。AMC23 使用 `math-ai/amc23` 的 40 道 test 题，固定 revision 为 `80815d37005feb82cd7f8fbc6901d5d3eff43057`；记录中的 `benchmark` 为 `amc23`，答案统一为整数字符串。现有评测命令会自动纳入 AMC23，并在 `.summary.json` 中按 benchmark 分别输出 Mean@8。

题目已随仓库提供，无需重新下载。从尚未包含 AMC23 的五项题库补齐时，可执行以下命令；重复执行不会重复添加，脚本会校验固定数据源的 SHA-256：

```bash
python scripts/prepare_amc23.py
```

以上评测命令明确使用 lenient 计分。归档的 `data/rq4.csv`、现有 RQ4 图和论文中的 Avg. 仍为原五项 benchmark 的实测平均分；AMC23 需要运行评测后才能纳入新的六项平均分。新增题目会改变评测缓存的数据集校验值，评测六项时请使用新的 `--output_file` 路径。复核历史报告时按该次运行原有的题集和计分协议比较。新环境下的耗时应以本次实测为准。

## 数据分析与绘图

绘制归档实验的数值只需 Python、NumPy 和 Matplotlib，Windows CPU 上也可执行。

```bash
python -m pip install numpy matplotlib
python analysis/plot_rq1_rq4.py
python analysis/plot_rq2.py
python analysis/plot_rq3.py
```

图输出到 `outputs/figures/`。RQ1 的 Top-KL decile 能量是逐 token **最后一层完整 hidden 向量**的梯度平方和占比。训练 KL 有每模型 106 个实测值，两个 token 占比指标有每模型 23 个实测值。RQ4 图使用 original FULL 基线。

完成两个规模的 paired FULL 后，可提取新运行的逐步数据和参数层附录：

```bash
python analysis/extract_signal.py
python analysis/plot_rq1_rq4.py --rq1 outputs/rq1.csv
python analysis/parameter_appendix.py
```

默认读取 `runs/1.7b/paired-full` 和 `runs/4b/paired-full`，分别输出 `outputs/rq1.csv` 和 `outputs/parameter_appendix/`。已有运行可通过 `--run-17` 和 `--run-4` 指定。归档版的 RQ2/RQ3 作图数据已随附在 `data/`。需要从合并报告重新提取这些数据时，执行：

```bash
python -m pip install lxml
python analysis/extract_report.py "/path/to/OPD_论文全部实验报告_合并版.zip"
```

## 论文

Overleaf 上传 `paper/` 的内容，主文件选 `main.tex`。本地有 TeX 环境时可执行：

```bash
cd paper
latexmk -pdf main.tex
```

## 实验代码索引

- RQ1：`runtime/verl/workers/actor/opd_top16.py`、`opd_token_audit_runtime.py`、`opd_token_audit.py`。
- RQ2：同目录的 `opd_gradient_scopes.py`、`opd_group_scopes.py` 和 `opd_replication17.py`。
- RQ3：同目录的 `opd_gradient_study.py`、`opd_gradient_study_math.py`，以及 `runtime/verl/trainer/ppo/opd_fixed_validation.py`。
- RQ4：`runtime/verl/workers/actor/opd_adaptive_stability.py` 实现三个 checkpoint 的停止窗口；`configs/<scale>/reuse.yaml` 中的 `jump_reuse_updates=2` 和 53 轮预算控制复用。
