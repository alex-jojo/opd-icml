# opd-icml

在四张 H200 上运行 OPD FULL 和 RQ4 策略对照，自动准备模型和数据、训练、评测并打包结果。

**模型范围**包括 DeepSeek-Coder 1.3B/6.7B/33B、Qwen3 1.7B/4B/8B 和 Gemma3 12B。总入口训练七组 FULL。四个数学模型和 DeepSeek-Coder 1.3B 各另做一次 rollout reuse，共 12 次训练。每组的早停成绩从各自 FULL 轨迹首次满足稳定条件的 checkpoint 取得。七组各自评测一次未训练学生模型，整套对比表共 26 行，包含七行 Base。

## 1. 下载代码、安装环境（首次执行）

在服务器分配给你的工作目录下执行。需要 Linux x86_64、可用的 NVIDIA 驱动、Conda，以及下载依赖和模型的网络访问。

```bash
git clone --branch branch-h200 https://github.com/alex-jojo/opd-icml.git
cd opd-icml/opd-icml

conda create -n opd-h200 python=3.10 -y
conda activate opd-h200
bash scripts/setup_h200.sh
```

项目目录有两层；执行后能看到 `train.py`、`configs/` 和 `scripts/` 即正确。安装在独立环境中；后续登录只需回到此目录并执行 `conda activate opd-h200`。

## 2. 登录模型账号、设置 W&B

完整七组包括 Gemma，需要 Hugging Face 账号已取得对应模型访问权限：

```bash
export HF_TOKEN='<YOUR_HF_TOKEN>'
```

在当前终端执行上述命令后，模型下载和训练脚本会自动读取 `HF_TOKEN`。重新登录服务器后需要重新执行。

直接执行以下命令设置 W&B，无需创建配置文件或交互输入：

```bash
export WANDB_API_KEY='<YOUR_WANDB_API_KEY>'
export WANDB_MODE=online
export WANDB_PROJECT=opd-icml
```

这些变量作用于当前终端及其启动的实验进程。重新登录后需重新设置。可选 `WANDB_ENTITY` 指定团队；只记录离线曲线用 `export WANDB_MODE=offline`，关闭 W&B 用 `export WANDB_MODE=disabled`。

## 3. 启动实验

### 一次跑完整套

建议在服务器已有的 tmux 会话中执行。确认分配的四张 GPU、主存和工作磁盘：

```bash
nvidia-smi
free -h
df -h .
```

直接运行下面这一条命令。假设分给你的 GPU 为 `0,1,2,3`：

```bash
bash scripts/h200/run_all.sh \
  --devices 0,1,2,3 \
  --output runs/h200-suite/run1
```

本命令使用下表中的七组学生与教师配对。每组用同一 GPU 配置评测 FULL 早停点。四个数学模型和 DeepSeek-Coder 1.3B 还会用该配置训练和评测复用策略。

| 模型配对（学生 → 教师） | 训练数据 | 测评 benchmark | 指标 | GPU 编号 | 运行顺序 |
|---|---|---|---|---|---|
| DeepSeek-Coder-1.3B-Base → 1.3B-Instruct | Eurus-2-RL-Data，Code 子集 | LiveCodeBench v6、HumanEval+ | pass@8 | 0 | 第一阶段：可并行 |
| Qwen3-1.7B-Base → Qwen3-1.7B | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 1 | 第一阶段：可并行 |
| Qwen3-4B-Base → Qwen3-4B | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 2 | 第一阶段：可并行 |
| DeepSeek-Coder-6.7B-Base → 6.7B-Instruct | Eurus-2-RL-Data，Code 子集 | LiveCodeBench v6、HumanEval+ | pass@8 | 3 | 第一阶段：可并行 |
| Qwen3-8B-Base → Qwen3-32B | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 0,1,2,3 | 第2阶段：独占四卡 |
| Gemma-3-12B-PT → Gemma-3-12B-IT | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 0,1,2,3 | 第3阶段：独占四卡 |
| DeepSeek-Coder-33B-Base → 33B-Instruct | Eurus-2-RL-Data，Code 子集 | LiveCodeBench v6、HumanEval+ | pass@8 | 0,1,2,3 | 第4阶段：独占四卡 |

表中 GPU 编号对应 `--devices 0,1,2,3` 且允许四组并行时的分配。前四组分别使用 GPU 0、1、2、3；后三组依次使用全部四张卡。若 CPU/主存预算使并行数下降，调度器会复用前面的 GPU 槽位，实际分配会在正式启动时自动打印；更换 `--devices` 时按传入顺序对应。

总入口自动准备模型和数据、调度各组并打包结果。整套从头运行时使用新的 `--output` 目录，例如第二次改为 `runs/h200-suite/run2`。总入口暂不支持中断后续训。模型缓存、checkpoint、合并模型和统计数据会占用较多磁盘，33B 训练还需要充分主存。H200 吞吐与显存峰值尚未实测。

### 单独跑一个模型或策略

在含 `train.py` 的目录中选一行执行。`--stage all` 会训练 FULL、评测终点和早停点、评测未训练的 Base，成功后把训练记录和评测结果一起打包。单卡示例使用 GPU 0，可将 `--devices` 改为实际分配的卡号。

| 学生模型 | 单独启动 FULL |
| --- | --- |
| DeepSeek-Coder 1.3B | `bash scripts/h200/train_eval_deepseek_coder_1_3b.sh --stage all --devices 0 --output runs/h200/deepseek13-full` |
| Qwen3 1.7B | `bash scripts/h200/train_eval_qwen3_1_7b.sh --stage all --devices 0 --output runs/h200/qwen17-full` |
| Qwen3 4B | `bash scripts/h200/train_eval_qwen3_4b.sh --stage all --devices 0 --output runs/h200/qwen4-full` |
| DeepSeek-Coder 6.7B | `bash scripts/h200/train_eval_deepseek_coder_6_7b.sh --stage all --devices 0 --output runs/h200/deepseek67-full` |
| Qwen3 8B | `bash scripts/h200/train_eval_qwen3_8b.sh --stage all --devices 0,1,2,3 --output runs/h200/qwen8-full` |
| Gemma3 12B | `bash scripts/h200/train_eval_gemma3_12b.sh --stage all --devices 0,1,2,3 --output runs/h200/gemma12-full` |
| DeepSeek-Coder 33B | `bash scripts/h200/train_eval_deepseek_coder_33b.sh --stage all --devices 0,1,2,3 --output runs/h200/deepseek33-full` |

单独跑 Reuse 时，在对应模型的命令中加 `--rq4-strategy reuse`，并换一个输出目录。整套实验包含 Qwen3 1.7B、4B、8B、Gemma3 12B 和 DeepSeek-Coder 1.3B 的 Reuse。以 Qwen3 1.7B 为例：

```bash
bash scripts/h200/train_eval_qwen3_1_7b.sh --stage all --devices 0 --rq4-strategy reuse --output runs/h200/qwen17-reuse
```

### 正式实验的训练与记录

七组 FULL 共用 `paired-full` 诊断配置，均做 106 次参数更新，全局 batch 为 128，学习率为 1e-5。FULL 每轮生成 128 条新回答并更新一次。Reuse 每批回答更新两次，53 轮共做 106 次更新。

每轮的训练输入、学生回答和评分写入 `training/training_records/<轮号>.jsonl`。每次参数更新抽样 128 个 token 记录隐藏层梯度统计。在更新 1、5、10、…、105、106，默认从该步样本中取 64 个 token 计算全模型参数梯度能量。逐步汇总在 `training/training_records/rq1_step_XXXX.json`，其中参数梯度能量只在观测步有值。数学组另记答案提取与判分，代码组另记代码测试与执行结果。各字段和评测文件见[实验保存说明](scripts/h200/实验保存说明.md)。

2026-11-21 的 Qwen3 1.7B 单卡 A100 训练包 `1.7b-qwen17-recorded-20step-20261121-121845-results.zip` 使用了单独的配置。它覆盖 20 次更新，在第 1、5、10、15、20 步各抽 32 个参数梯度 token，包含训练和固定验证记录。上面的 H200 FULL 命令采用 106 次更新、每个观测步 64 个参数梯度 token，并在训练后运行最终 benchmark 评测。

RQ2 在第 1、5、10、…、105、106 次 FULL 参数更新后测量同一组固定 probe。`training/training_records/rq2/` 保存初始化的 hidden 与位置、各观测步的累计位移 RMS、特征值谱和前 128 个方向。从第 10 步起还记录固定第 10 步子空间的覆盖率和重合度。
同目录的 `gradient_XXX.npz` 在上述 23 个观测点保存训练 token 的完整 hidden 梯度谱及其自身 top-128 能量占比。第 15 次更新起，同时保存梯度在第 10 次更新后固定的 probe 子空间 U₁₀ 内的能量占比。

RQ3 的 `training/training_records/rq3/checkpoint_XXX.json` 按更新前的检查点编号。0、10、…、100 各存 128 条验证回答在下一次更新前后的配对 KL；10、30、60、100 另存图 7(b、c) 的六对子集明细和均值。

RQ4 比较结果在整个实验 ZIP 的 `rq4_comparison.json`。各规模内的策略共用数据划分和评测协议。FULL 每 5 步检查早停条件，首次触发时单独保存权重并继续训练到第 106 步。早停耗时记录从 FULL 启动到该权重保存完成的累计时间。复用在 53 轮中做 106 次更新，并在 11 个观测轮记录同批回答第二次更新的 KL 降幅。数学汇总五项 Mean@8；代码汇总 LiveCodeBench、HumanEval+ 的 pass@8，各自按题集等权平均。逐轮文件和单组命令见 [H200 说明](scripts/h200/README.md#rq4-策略对照)。

## 4. 看进度、下载结果

在另一个终端、同一项目目录下：

```bash
cat runs/h200-suite/run1/status.json
tail -F runs/h200-suite/run1/4b/logs/train.log
```

单模型运行完成后，下载终端 `[DOWNLOAD ZIP]` 打印的文件。例如，上面的 Qwen3 1.7B FULL 命令生成 `runs/h200/1.7b-qwen17-full-results.zip`，包内的 `qwen17-full/eval/final_mean8.summary.json` 是终点成绩。

整套运行全部成功后，下载以下文件：

```text
runs/h200-suite/run1-results.zip
```

ZIP 包含训练统计、表征和参数梯度数据、生成回答、评测结果、配置和日志。模型权重仅用于服务器端合并与评测，默认在该组评测成功后清理。需要保留权重时，启动命令加 `--keep-model-files`。

---

- [单模型运行、调参、重试评测和资源设置](scripts/h200/README.md)
- [实验协议、历史配置、数据分析和论文说明](docs/REFERENCE.md)


重型诊断改为每 5 次参数更新，另保留首尾及原有验证诊断点；轻量 RQ1 逐步记录。完整调度与缺测字段见[重型诊断频率](scripts/h200/README.md#重型诊断频率)。
