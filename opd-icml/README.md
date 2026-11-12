# opd-icml

四张 H200 上运行七组 OPD FULL 实验：自动准备模型和数据、训练并记录统计、评测、打包结果。

**范围：** DeepSeek-Coder 1.3B/6.7B/33B，Qwen3 1.7B/4B/8B，Gemma3 12B。此入口不自动运行 Base 基线、Adaptive early stop 或 Rollout reuse。

## 1. 下载代码、安装环境（首次执行）

在服务器分配给你的工作目录下执行。需要 Linux x86_64、可用的 NVIDIA 驱动、Conda，以及下载依赖和模型的网络访问。

```bash
git clone https://github.com/alex-jojo/opd-icml.git
cd opd-icml/opd-icml

conda create -n opd-h200 python=3.10 -y
conda activate opd-h200
bash scripts/setup_h200.sh
```

项目目录有两层；执行后能看到 `train.py`、`configs/` 和 `scripts/` 即正确。安装在独立环境中；后续登录只需回到此目录并执行 `conda activate opd-h200`。

## 2. 登录模型账号、设置 W&B

完整七组包括 Gemma，需要 Hugging Face 账号已取得对应模型访问权限：

```bash
huggingface-cli login
```

直接执行以下命令设置 W&B，无需创建配置文件或交互输入：

```bash
export WANDB_API_KEY='在这里粘贴你的 W&B API key'
export WANDB_MODE=online
export WANDB_PROJECT=opd-icml
```

这些变量作用于当前终端及其启动的实验进程。重新登录后需重新设置。可选 `WANDB_ENTITY` 指定团队；只记录离线曲线用 `export WANDB_MODE=offline`，关闭 W&B 用 `export WANDB_MODE=disabled`。

## 3. 启动整套实验

建议在服务器已有的 tmux 会话中执行。确认分配的四张 GPU、主存和工作磁盘：

```bash
nvidia-smi
free -h
df -h .
```

先查看计划，再正式运行。下面假设分给你的 GPU 为 `0,1,2,3`：

```bash
bash scripts/h200/run_all.sh --devices 0,1,2,3 --dry-run

bash scripts/h200/run_all.sh \
  --devices 0,1,2,3 \
  --output runs/h200-suite/run1
```

本命令运行以下七组实验（模型列为学生 → 教师）：

| 模型配对（学生 → 教师） | 训练数据 | 测评 benchmark | 指标 | GPU 数 | 运行顺序 |
|---|---|---|---|---|---|
| DeepSeek-Coder-1.3B-Base → 1.3B-Instruct | Eurus-2-RL-Data，Code 子集 | LiveCodeBench v6、MBPP+、HumanEval+ | pass@8 | 1 | 第一阶段：可并行 |
| Qwen3-1.7B-Base → Qwen3-1.7B | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 1 | 第一阶段：可并行 |
| Qwen3-4B-Base → Qwen3-4B | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 1 | 第一阶段：可并行 |
| DeepSeek-Coder-6.7B-Base → 6.7B-Instruct | Eurus-2-RL-Data，Code 子集 | LiveCodeBench v6、MBPP+、HumanEval+ | pass@8 | 1 | 第一阶段：可并行 |
| Qwen3-8B-Base → Qwen3-32B | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 4 | 第2阶段：独占四卡 |
| Gemma-3-12B-PT → Gemma-3-12B-IT | DAPO14k | MATH500、HMMT25、HMMT26、AIME26、Minerva、AMC23 | Mean@8 | 4 | 第3阶段：独占四卡 |
| DeepSeek-Coder-33B-Base → 33B-Instruct | Eurus-2-RL-Data，Code 子集 | LiveCodeBench v6、MBPP+、HumanEval+ | pass@8 | 4 | 第4阶段：独占四卡 |

前四组各用单卡，按 CPU/主存预算最多四组并行；后三组每组独占四卡，依次运行。

**不用逐个下载模型、准备数据或启动七个脚本。**总入口会自动完成这些步骤。模型缓存、checkpoint、合并模型和统计数据会占用较多磁盘；33B 训练还需要充分主存。

七组均为 `paired-full`，106 轮、batch 128、学习率 1e-5。参数梯度诊断默认每个观测点抽样 256 个 token，计算各 token 的全模型参数梯度并估计总体能量；hidden 统计同时保留。Code 评测 pass@8，Math 评测六项 Mean@8。H200 吞吐与显存峰值尚未实测。

输出目录必须是新目录；再次从头运行换成 `run2`。当前入口不支持中断后的训练续跑。

## 4. 看进度、下载结果

在另一个终端、同一项目目录下：

```bash
cat runs/h200-suite/run1/status.json
tail -F runs/h200-suite/run1/4b/logs/train.log
```

全部成功结束后，在服务器文件浏览器下载：

```text
runs/h200-suite/run1-results.zip
```

ZIP 包含训练统计、表征和参数梯度数据、生成回答、评测结果、配置和日志。权重保留在各模型的 `training/checkpoints/` 和 `merged/`，不包含在 ZIP 中。

---

- [单模型运行、调参、重试评测和资源设置](scripts/h200/README.md)
- [实验协议、历史配置、数据分析和论文说明](docs/REFERENCE.md)

