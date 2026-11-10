# DeepSeek-Coder 1.3B Base → Instruct：单卡 A100 / 四卡 Code OPD

学生为 `deepseek-ai/deepseek-coder-1.3b-base`（revision `c919139c3a9b4070729c8b2cca4847ab29ca8d94`），教师为 `deepseek-ai/deepseek-coder-1.3b-instruct`（revision `e063262dac8366fc1f28a4da0ff3c50ea66259ca`）。训练使用 Eurus code 子集，协议为学生采样、缓存教师目标的 conditional student-support Top16 reverse-KL，全参数更新学生，教师冻结。当前使用一张 A100 80GB，指定 `--hardware a100x1`：GPU 数、rollout TP 和梯度诊断 world size 均为 1。四卡 L40/L40S 保留 `--hardware l40x4`；这里不使用测评的 `--parallel-mode data`。

这是新接入的模型配对。CPU 测试验证了原权重不变、tokenizer/EOS 对齐、Llama 梯度与配置；尚未执行这组模型的完整 GPU OPD 训练。实际显存与耗时以训练服务器为准。

## 1. 进入项目，补齐训练依赖

当前服务器已有 `code-eval310` 测评环境，直接增量安装，不创建第二份 PyTorch 环境：

```bash
cd /workspace/opd-icml/opd-icml
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate code-eval310
bash scripts/setup_opd_training.sh
```

新 A100 服务器没有现成环境时，先将更新后的项目放到服务器，再从项目目录执行：

```bash
conda create -n code-eval310 python=3.10 pip -y
conda activate code-eval310
bash scripts/setup_code_eval.sh
bash scripts/setup_opd_training.sh
```

已有同名环境时直接激活并补齐依赖，不重复创建。安装固定 Ray 2.46.0、Hydra 1.3.2、PEFT 0.15.2、Accelerate 1.7.0、Liger 0.5.10、TensorDict 0.8.3 等，与 Torch 2.6/cu124、vLLM 0.8.5.post1 共用。安装注意力库使用已核对的对应 ABI 预编译 wheel，不在服务器编译。

原实例磁盘总共 32 GB，检查时仅余约 7.9 GB。需容纳约 2.7 GB 的 Instruct、训练集、训练依赖、诊断和 checkpoint；仅保存 FP32 学生权重就可能约 5.4 GB。下载和正式训练前先准备足够磁盘空间，例如新建容器盘 150 GB 的实例；该容量是本实验起始预算，多个模型/多次运行需要更多。Vast 容器盘创建后不能直接调整大小，备份后迁移到新实例。`/workspace` 没有 volume 时不保证在销毁后保留。

## 2. 准备训练集

```bash
python scripts/prepare_eurus_code.py
```

得到 `data/eurus_code_train.parquet`（25276 题）及 `data/eurus_code_validation.parquet`（1024 题）。读取固定源版本，保留代码测试用例；不是先前的三个测评题集。默认通过远程 Parquet 分块读取，不要求另存整份混合训练集。已有原始源文件时可用 `--source-dir`，新数据盘可配 `--output-dir`，训练时用 `--data`、`--val-data` 指定对应文件。

Code 配置将 `data.expected_filtered_train_rows` 显式设为 `null`，因为长度过滤后的数量取决于模型 tokenizer；不沿用数学任务的 13534 条断言。准备阶段仍严格校验上述固定源数据行数、测试用例和文件校验值。当前 106 轮是固定训练预算，不表示完整遍历一遍 Eurus Code。

## 3. 准备学生与教师

新服务器还没有模型时，自动下载 Base 学生和 Instruct 教师：

```bash
python scripts/prepare_models.py --scale deepseek-coder-1.3b
```

原始 Base 已下载或已迁移到新服务器时，可在准备和训练命令中都加 `--student models/deepseek-coder-1.3b-base` 复用权重；该目录必须存在，不要在空的新服务器指定它。教师自动按固定 revision 下载，只取 safetensors，避免同时下载 `.bin` 和 safetensors 两份权重。已有本地 Instruct 时在准备和训练命令中都额外传 `--teacher /path/to/deepseek-coder-1.3b-instruct`。

Base 和 Instruct 的词表/token ID 完全一致，但 Base EOS=32014、Instruct EOS=32021（`<|EOT|>`），Base 没有 chat template。准备入口创建独立学生训练目录 `models/deepseek-coder-1.3b/student-opd-eos-v2/`：链接原 Base 权重，共用 Instruct tokenizer/template；主 EOS 仍为 `<|EOT|>`，但 `generation_config.json` 和 rollout 显式停止条件同时接受 `[32021, 32014]`。响应 mask 按引擎返回的真实长度构造，包含最后一个 EOS，排除所有右侧 padding，正确处理 Base EOS 与 PAD 同为 32014 的情况。原 Base、教师及旧 `student-opd/` 目录不覆盖；链接支持硬链接或符号链接，不复制或量化权重，源目录必须保持可用。

## 4. 查看配置并启动单卡 A100 80GB OPD

专用入口为 `scripts/train_deepseek_code_a100.sh`。已激活 `code-eval310` 后，可直接编辑该文件顶部的变量，并执行：

```bash
bash scripts/train_deepseek_code_a100.sh --dry-run
bash scripts/train_deepseek_code_a100.sh
```

`ROLLOUT_BATCH_SIZE=128` 是每轮采样题数；`MINI_BATCH_SIZE=128` 是一次参数更新覆盖的全局题数，当前 OPD 协议要求两者相同；`MICRO_BATCH_SIZE=1` 是每张 GPU 一次前向/反向的题数。单卡 128/128/1 通过 128 个 micro-batch 累积后更新一次。调大 micro-batch 影响显存和计算效率；改变 rollout/mini-batch 则改变每次更新的数据量，106 轮对应的总训练题数也会改变。

脚本同时提供 `LEARNING_RATE=1e-5`、`ROLLOUT_MAX_NUM_SEQS=16`（生成并发）、`ROLLOUT_GPU_MEMORY_UTILIZATION=0.30`、`PARAMETER_TOKEN_SAMPLES=256`、`CODE_WORKERS=8` 及模型/输出路径。默认保持现有实验设置；脚本参数在硬件预设之后生效。模型的双 EOS 仍由准备入口生成，训练轮数和其他诊断仍继承 YAML。

脚本还支持 `NUM_GPUS=1`（默认）与 `ROLLOUT_TP_SIZE`（默认等于卡数）。例如 `NUM_GPUS=4 bash scripts/train_deepseek_code_a100.sh --dry-run` 会设置四卡 FSDP、rollout TP=4 和诊断 world size=4；`NUM_GPUS=4 ROLLOUT_TP_SIZE=2` 可拆成两组生成 worker。全局 batch、学习率、回答上限、256-token 抽样和 106 轮预算保持不变。每张卡必须能容纳所选 micro-batch；单卡机器只能用 `--dry-run` 检查多卡配置，不能运行多卡任务。

逐 token 参数诊断会在 GPU 内完成范数累加，减少小标量同步；对因果模型只重算目标 token 的前缀，实际及未裁剪前向 token 数写入诊断 JSON。原训练前向/反向及采样种子不变。数值等价性由微型模型检查，实际 A100/多卡提速需实测，改变卡数或批处理不保证逐位相同。原运行进程继续使用已加载代码，更新后由用户另选输出目录启动。

也可用环境变量临时覆盖，先查看生效配置：

```bash
MICRO_BATCH_SIZE=2 ROLLOUT_MAX_NUM_SEQS=32 \
  OUTPUT_DIR=runs/code/deepseek-coder-1.3b/a100-micro2-seqs32 \
  bash scripts/train_deepseek_code_a100.sh --dry-run
```

此命令只检查配置；更大 micro-batch 或生成并发的显存峰值和速度需要另行实测。正式运行时去掉 `--dry-run`，使用新的输出目录。当前进程不会因编辑脚本而改变。

对应的 Python 入口仍可直接使用：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

python train.py --scale deepseek-coder-1.3b --task code \
  --experiment paired-full --hardware a100x1 \
  --parameter-token-gradients sampled --parameter-token-samples 256 \
  --output runs/code/deepseek-coder-1.3b/paired-full-a100x1-sample256-eosv2 \
  --dry-run

python train.py --scale deepseek-coder-1.3b --task code \
  --experiment paired-full --hardware a100x1 \
  --parameter-token-gradients sampled --parameter-token-samples 256 \
  --output runs/code/deepseek-coder-1.3b/paired-full-a100x1-sample256-eosv2
```

最多 106 轮，每轮一次学生更新，全局 rollout batch=128，prompt 上限 2048、response 上限 3072。单卡 micro-batch=1，通过梯度累积覆盖全局 batch；启用 checkpointing、optimizer CPU offload 和 teacher parameter CPU offload，rollout 并发上限 16。保留每轮训练记录与 hidden/参数梯度诊断；同一个输出目录已有内容时拒绝覆盖。若仍用原四卡服务器，将 `CUDA_VISIBLE_DEVICES` 改为 `0,1,2,3`、`--hardware` 改为 `l40x4`，并另选输出目录。

配置与 Hydra 测试不等于 GPU 显存验证；尚未在 A100 上完成这组模型的正式训练。不要只改 `CUDA_VISIBLE_DEVICES=0` 却继续用 `l40x4` 或原始默认配置，它们会继续申请四卡。

**当前 DeepSeek Code `paired-full` 默认采用用户确认的 256-token 分层抽样诊断。** 每个诊断点，从整批有效 token 的 KL 最高 10% 中随机抽 64 个、其余 90% 中随机抽 192 个，层内无放回，随机种子 42 并随 step/inner 派生；小总体会减少抽样量或全部测量。第 1 步、每 5 次更新和末步的测量频率不变。训练仍使用所有有效 token 的 loss，训练 batch 和优化器更新方式不变。

每个抽中 token 的全部可训练参数梯度仍精确计算，按层总体数/抽样数加权估计梯度能量总量；占比采用估计总量之比，不能称为全量精确占比或无偏占比。日志保存实际样本 ID、权重、种子、层规模和近似 95% 区间。256-token 诊断按之前实测速率约需 3 分钟，仅是该项额外计算的估计，整轮还包含生成、判题和其他诊断。

全量 hidden 统计继续保存；同一批抽样 token 的 hidden/参数能量、KL 和权重另存 `replication/token_audit/parameter_sample_<step>_inner_1.npz`。JSON 的 `paired_parameter_hidden_sample` 用相同样本及权重比较两种能量。CSV 明确标记 sampled、样本数和总体数；未观测到的非空组占比留空，不填零。参数能量自身排名的全量 concentration 曲线不由这个样本伪造。

显式指定 `--parameter-token-gradients exact-all` 可回到全量精确模式，`off` 可关闭额外逐 token 参数诊断；这两种模式均保留原有 hidden 统计。模式或样本预算改变时使用新输出目录。当前正在运行的旧进程不会自动切换；由用户停止后使用新命令重新启动。

## 5. 训练后评测与可比性

本轮训练使用 Instruct 的提示格式。要比较 OPD 的净增益，应先用准备后的、尚未更新的 `student-opd-eos-v2` 与训练后合并 checkpoint，在同样的 `--prompt-style chat`、题集、双 EOS 停止规则和采样参数下各测一次。此前原始 Base 的 `--prompt-style completion` 成绩属于另一提示协议，不是严格同提示的训练前基线。

checkpoint 合并与 code pass@4 命令沿用 README 的流程，`--model-path` 指向合并后的学生模型。新增模型默认提供 paired-full、original-full、early-stop、reuse 的 math/code 配置；这次教师针对代码任务，正式实验请明确使用 `--task code`。
