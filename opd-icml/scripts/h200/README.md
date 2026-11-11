# H200 七组训练与评测

从零配置环境、准备模型和数据，以及七组完整启动命令，统一见项目主 [README 的 H200 操作指南](../../README.md#h200-操作指南)。本页保留脚本索引和使用概要。

在 Linux H200 服务器的项目根目录运行。先激活已安装项目依赖的 Python 3.10 环境；依赖安装沿用 `scripts/setup_code_eval.sh`、`scripts/setup_opd_training.sh`，代码评测还需 `scripts/setup_code_benchmarks.sh`。这些入口不安装软件。

按**学生模型规模**固定卡数，训练、rollout 和评测生成使用同一组卡。四卡评测采用 TP=4，单卡采用 TP=1。代码测试用例判题仍由 CPU 执行。

- `train_eval_deepseek_coder_1_3b.sh`：1.3B Base → 1.3B Instruct，单卡，code。
- `train_eval_deepseek_coder_6_7b.sh`：6.7B Base → 6.7B Instruct，单卡，code。
- `train_eval_deepseek_coder_33b.sh`：33B Base → 33B Instruct，四卡，code。
- `train_eval_qwen3_1_7b.sh`：1.7B Base → 1.7B，单卡，math。
- `train_eval_qwen3_4b.sh`：4B Base → 4B，单卡，math。
- `train_eval_qwen3_8b.sh`：8B Base → 32B 教师，四卡，math。
- `train_eval_gemma3_12b.sh`：12B PT → 12B IT，四卡，math。

例如，先检查命令，再正式运行：

```bash
bash scripts/h200/train_eval_qwen3_8b.sh --dry-run
bash scripts/h200/train_eval_qwen3_8b.sh
```

七个完整入口如下，按需选择，四卡实验需独占四张卡：

```bash
bash scripts/h200/train_eval_deepseek_coder_1_3b.sh
bash scripts/h200/train_eval_deepseek_coder_6_7b.sh
bash scripts/h200/train_eval_deepseek_coder_33b.sh
bash scripts/h200/train_eval_qwen3_1_7b.sh
bash scripts/h200/train_eval_qwen3_4b.sh
bash scripts/h200/train_eval_qwen3_8b.sh
bash scripts/h200/train_eval_gemma3_12b.sh
```

单卡默认使用 GPU 0，四卡默认使用 0,1,2,3。已有 `CUDA_VISIBLE_DEVICES` 时按其顺序取所需卡数，并传给训练及评测；不足时报错。也可明确选卡，`--devices` 必须正好提供所需数量：

```bash
bash scripts/h200/train_eval_qwen3_4b.sh --devices 2
bash scripts/h200/train_eval_qwen3_8b.sh --devices 0,1,2,3
```

公共流程在 `scripts/run_h200_experiment.py`：准备默认训练数据和缺失的代码 benchmark → 按固定 revision 准备模型并训练 → 检查最终 checkpoint 步数、world size 和全部模型分片 → 合并 HF 权重 → 评测。Gemma 需要已获得模型访问权限。已有本地权重可传 `--student`、`--teacher`；DeepSeek 自动复用 Instruct 模板和双 EOS，Gemma 自动提取文本权重。新增 6.7B/33B 配置目前提供本流程使用的 `paired-full`。

默认采用 `paired-full`、全局 batch/mini-batch=128、106 轮、每轮一次更新、学习率 1e-5、每卡 micro-batch=1。逐 token 参数梯度沿用现有启动脚本的抽样设置：每次 256 个 token（64 high-KL + 192 remaining），hidden 统计保留。需要全量时显式加 `--parameter-token-gradients exact-all`。H200 预设保留梯度检查点、optimizer 阶段 offload、teacher 参数 offload，rollout 并发 16、显存比例 0.30；这是保守起始配置，不代表已验证的最优吞吐。

DeepSeek 默认评测 LiveCodeBench、MBPP+、HumanEval+ 的 pass@4；Qwen/Gemma 默认评测 `data/benchmarks.jsonl` 的数学 Mean@8。训练后的 DeepSeek 使用 chat 协议。可传 `--task code` / `--task math` 切换任务，卡数规则不变；评测采样参数和长度沿用已有协议。默认评测并发 64，可用 `--eval-max-num-seqs` 调整并发，不改变采样数或最大生成长度。

默认输出为 `runs/h200/<scale>/<task>-paired-full/`，包含 `training/`、`merged/`、`eval/`、`logs/`。重新训练需用新的 `--output`。训练和评测可分两次执行，第二次必须使用相同输出目录和卡数：

```bash
bash scripts/h200/train_eval_qwen3_8b.sh --stage train --output runs/h200/qwen8-seed42
bash scripts/h200/train_eval_qwen3_8b.sh --stage eval --output runs/h200/qwen8-seed42
```

`--stage eval` 不重新训练，已验证的合并结果会复用，评测沿用原入口的断点缓存。合并失败留下的 `merged/` 没有完成标记时会拒绝复用；先将该目录移到其他位置，再重试。`--dry-run` 只输出命令和实际解析的训练配置，不下载、不创建运行目录、不使用 GPU。

6.7B 单卡和 33B 四卡仍需在真实 H200 上验证完整训练更新及诊断的显存峰值；脚本不会自动增加卡数、降低精度、减少诊断或改变实验预算。33B 合并和梯度诊断也需要足够的 CPU 内存及磁盘。当前本地检查验证入口、配置和流程衔接，不代表 H200 实跑通过。
