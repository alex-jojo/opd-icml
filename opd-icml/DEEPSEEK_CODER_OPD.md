# DeepSeek-Coder 1.3B Base → Instruct：四卡 Code OPD

学生为 `deepseek-ai/deepseek-coder-1.3b-base`（revision `c919139c3a9b4070729c8b2cca4847ab29ca8d94`），教师为 `deepseek-ai/deepseek-coder-1.3b-instruct`（revision `e063262dac8366fc1f28a4da0ff3c50ea66259ca`）。训练使用 Eurus code 子集，协议为学生采样、缓存教师目标的 conditional student-support Top16 reverse-KL，全参数更新学生，教师冻结。这里不使用测评的 `--parallel-mode data`；训练采用四卡 FSDP 和 rollout TP=4。

这是新接入的模型配对。CPU 测试验证了原权重不变、tokenizer/EOS 对齐、Llama 梯度与配置；尚未执行这组模型的完整 GPU OPD 训练。实际显存与耗时以训练服务器为准。

## 1. 进入项目，补齐训练依赖

当前服务器已有 `code-eval310` 测评环境，直接增量安装，不创建第二份 PyTorch 环境：

```bash
cd /workspace/opd-icml/opd-icml
source /root/miniforge3/etc/profile.d/conda.sh
conda activate code-eval310
bash scripts/setup_opd_training.sh
```

新服务器尚未配置时，先按 `DEEPSEEK_CODER_EVAL.md` 建立 Python 3.10 环境并运行 `scripts/setup_code_eval.sh`，再运行上面的训练依赖入口。安装固定 Ray 2.46.0、Hydra 1.3.2、PEFT 0.15.2、Accelerate 1.7.0、Liger 0.5.10、TensorDict 0.8.3 等，与 Torch 2.6/cu124、vLLM 0.8.5.post1 共用。安装注意力库使用已核对的对应 ABI 预编译 wheel，不在服务器编译。

原实例磁盘总共 32 GB，检查时仅余约 7.9 GB。需容纳约 2.7 GB 的 Instruct、训练集、训练依赖、诊断和 checkpoint；仅保存 FP32 学生权重就可能约 5.4 GB。下载和正式训练前先准备足够磁盘空间，例如新建容器盘 150 GB 的实例；该容量是本实验起始预算，多个模型/多次运行需要更多。Vast 容器盘创建后不能直接调整大小，备份后迁移到新实例。`/workspace` 没有 volume 时不保证在销毁后保留。

## 2. 准备训练集

```bash
python scripts/prepare_eurus_code.py
```

得到 `data/eurus_code_train.parquet`（25276 题）及 `data/eurus_code_validation.parquet`（1024 题）。读取固定源版本，保留代码测试用例；不是先前的三个测评题集。默认通过远程 Parquet 分块读取，不要求另存整份混合训练集。已有原始源文件时可用 `--source-dir`，新数据盘可配 `--output-dir`，训练时用 `--data`、`--val-data` 指定对应文件。

## 3. 准备学生与教师

原始 Base 已用于测评，可直接复用；只需补下载 Instruct：

```bash
python scripts/prepare_models.py \
  --scale deepseek-coder-1.3b \
  --student models/deepseek-coder-1.3b-base
```

教师自动按固定 revision 下载，只取 safetensors，避免同时下载 `.bin` 和 safetensors 两份权重。已有本地 Instruct 时额外传 `--teacher /path/to/deepseek-coder-1.3b-instruct`。

Base 和 Instruct 的词表/token ID 完全一致，但 Base EOS=32014、Instruct EOS=32021（`<|EOT|>`），Base 没有 chat template。准备入口创建独立学生训练目录 `models/deepseek-coder-1.3b/student-opd/`：链接原 Base 权重，共用 Instruct tokenizer/template，将运行时 EOS 设为 `<|EOT|>`，不改写或量化权重，也不修改原 Base 测评目录。链接支持硬链接或符号链接，不静默复制整份权重；源目录必须保持可用。

## 4. 查看配置并启动四卡 OPD

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

python train.py --scale deepseek-coder-1.3b --task code \
  --experiment paired-full --hardware l40x4 \
  --student models/deepseek-coder-1.3b-base \
  --parameter-token-gradients exact-all \
  --output runs/code/deepseek-coder-1.3b/paired-full \
  --dry-run

python train.py --scale deepseek-coder-1.3b --task code \
  --experiment paired-full --hardware l40x4 \
  --student models/deepseek-coder-1.3b-base \
  --parameter-token-gradients exact-all \
  --output runs/code/deepseek-coder-1.3b/paired-full
```

最多 106 轮，每轮一次学生更新，全局 rollout batch=128，prompt 上限 2048、response 上限 3072。四卡 micro-batch=1，启用 checkpointing、optimizer CPU offload 和 teacher parameter CPU offload。保留每轮训练记录与 hidden/参数梯度诊断；同一个输出目录已有内容时拒绝覆盖。

**`exact-all` 保留用户指定的全部有效 token 的精确参数梯度。第 1 步就会进入昂贵的诊断，每个被测 token 额外做一次完整学生 forward/backward；看到 `[parameter tokens]` 进度不是卡死。** 之后每 5 次更新及末步继续记录。只为快速检查训练流程时，可明确将 `exact-all` 改成 `off` 并使用另一个输出目录；正式梯度实验使用默认开启模式。不会因模型较小或使用四卡而自动降为抽样。

## 5. 训练后评测与可比性

本轮训练使用 Instruct 的提示格式。要比较 OPD 的净增益，应先用准备后的、尚未更新的 `student-opd` 与训练后合并 checkpoint，在同样的 `--prompt-style chat`、题集和采样参数下各测一次。此前原始 Base 的 `--prompt-style completion` 成绩属于另一提示协议，不是严格同提示的训练前基线。

checkpoint 合并与 code pass@4 命令沿用 README 的流程，`--model-path` 指向合并后的学生模型。新增模型默认提供 paired-full、original-full、early-stop、reuse 的 math/code 配置；这次教师针对代码任务，正式实验请明确使用 `--task code`。
