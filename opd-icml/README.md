# opd-icml

Qwen3-1.7B / 4B on-policy distillation 实验代码。所有命令以此目录为项目根目录。训练采用本目录随附的修改版 verl。绘图输入与论文源码可以单独使用。

| 目录或入口 | 内容 |
|---|---|
| `train.py`、`configs/{1.7b,4b}/` | 训练入口、各实验的 Hydra 配置、学生和教师模型的固定 revision |
| `runtime/verl/workers/actor/` | Top-16 OPD、token 诊断、表征谱、梯度一致性和表征稳定性停止判据 |
| `runtime/verl/trainer/ppo/` | rollout、优化更新、复用、验证集与 checkpoint 管理 |
| `scripts/` | 安装 GPU 环境、准备 DAPO14k、Mean@8 数学评测 |
| `analysis/`、`data/` | 结果提取、作图、参数层附录，以及绘图输入和六项 benchmark 题目 |
| `paper/main.tex` | 论文入口及当前正文和附录引用的文件 |

## 训练配置

| `--experiment` | 论文用途 | 新 rollout 轮数 | 每轮更新 | 原运行 GPU 数（1.7B / 4B） |
|---|---|---:|---:|---:|
| `paired-full` | RQ1–RQ3。配对研究中的 FULL，使用 128 道排除训练的验证题 | 106 | 1 | 2 / 2 |
| `original-full` | RQ4 耗时与 benchmark 的 FULL 基线 | 106 | 1 | 1 / 2 |
| `early-stop` | RQ4 Adaptive early stop，每五步检查一次稳定性 | 上限 106，原运行在 40 / 50 步停止 | 1 | 1 / 2 |
| `reuse` | RQ4 Rollout reuse，按相同优化更新预算比较 | 53 | 2 | 1 / 2 |

配置来自成功运行的训练日志中实际解析的参数。paired FULL 的 rollout batch 是 128；原始研究的 batch 是 160，使用其原有的 128/32 协议。两套协议分别有自己的 FULL 参考结果。模型和数据路径在运行时注入，日志默认输出到控制台。启动入口用于新实验，每个配置使用独立输出目录。

## 安装和准备数据

在 Linux CUDA 训练环境中，从本目录执行。安装脚本使用 Python 3.10、PyTorch 2.6.0 / cu124、vLLM 0.8.5.post1 和原有项目的依赖版本。

```bash
bash scripts/setup.sh
source .venv/bin/activate
python scripts/prepare_dapo14k.py --output data/dapo14k_train.parquet
```

默认使用模型配置中记录的 Hugging Face revision。已有模型可通过 `--student /path/to/student --teacher /path/to/teacher` 指定。训练数据可通过 `--data` 指定。首次下载模型和训练集需要网络。

## 启动实验

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
