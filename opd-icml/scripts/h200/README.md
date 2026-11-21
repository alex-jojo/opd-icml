# H200 高级运行参考

首次部署和完整七组启动见[快速开始](../../README.md)。以下命令仅在单独运行、调参或恢复评测时使用；均在已激活的 Python 3.10 环境中执行。

所有模型、数据、配置、输出和 ZIP 路径均以项目根目录（含 `train.py` 的目录）为基准；例如 `models/student`、`data/benchmarks.jsonl`、`runs/h200-suite/run1`。入口会自行定位项目根目录，从其他工作目录启动也适用。项目外资源可写为 `../shared/...`，运行计划与新生成的路径标记保存相对路径。

所有入口继承当前终端的 `WANDB_API_KEY`、`WANDB_MODE`、`WANDB_PROJECT` 和可选的 `WANDB_ENTITY`。无需配置文件；设置方法见[快速开始](../../README.md#2-登录模型账号设置-wb)。

默认七组的输出目录、逐步记录、评测文件和清理时机见[实验保存说明](./实验保存说明.md)。

## 整套实验

```bash
# 新目录：准备数据/模型、训练、合并、生成、判题。
bash scripts/h200/run_all.sh --devices 0,1,2,3 --output runs/h200-suite/run1

# 只选部分模型；仍指定服务器上的四张卡。
bash scripts/h200/run_all.sh --devices 0,1,2,3 \
  --scales 1.7b 4b 8b --output runs/h200-suite/qwen

# 恢复原来的保守批处理与卸载设置。
bash scripts/h200/run_all.sh --profile conservative --output runs/h200-suite/conservative
```

七组模型的 paired-full 基线均为 106 轮、每轮一次更新、全局 batch/mini-batch=128、学习率 1e-5。七组模型均在各自的 FULL 训练中保存早停权重，各另跑一次复用训练。默认在第 1、5、10、…、105、106 次参数更新前抽样 256 个 token（64 个来自缓存 KL 最高的十分位，另 192 个来自其余位置），分别计算全模型参数梯度能量。有效位置少于 256 个时全部测量。`--parameter-token-gradients exact-all` 可显式切换到原全量观测模式。

每次更新的 RQ1 记录位于 `training/training_records/rq1_step_XXXX.json`，随结果 ZIP 导出。`thresholds.kl_p90` 是第 1 步全部有效 token 的 KL 第 90 百分位数，后续步沿用它。`groups.kl_above_step1_p90.fraction` 是当前训练 KL 达到该固定阈值的比例。`groups.kl_top10.quantities` 分别给出整批 hidden 梯度能量占比和抽样估计的参数梯度能量占比。`samples` 中的列按 token 对齐，包含回答行、位置、token ID、训练 KL、能量和权重。复算参数占比时，用 `sampling_weight * parameter_gradient_energy` 得到每条样本的加权能量，再用 `in_current_kl_top10` 选中项的和除以全部项的和。`sampling_stratum` 标记抽样时的层，成员判定使用本步训练 KL。

RQ2 的 `training/training_records/rq2/probe_initial.npz` 保存固定回答的输入、掩码、位置及初始化 hidden。`row` 和 `response_position` 标识每行 hidden 所预测的回答 token。FULL 在第 1、5、10、…、105、106 次更新后生成对应的 `step_XXX.npz`，共 23 个观测点。各步基于同一 probe，`raw_displacement_rms` 和 `unit_displacement_rms` 均以初始化为参照。归一化位移的二阶矩按降序保存 `eigenvalues`；`cumulative_energy_coverage[r-1]` 给出前 r 个方向的能量占比。`basis` 的列是当前前 128 个方向，任意两个检查点的重合度可用 `np.linalg.norm(Ua.T @ Ub, 'fro')**2 / Ua.shape[1]` 复算。

第 10 步冻结 RQ2 参考基。从该步起，`fixed_u10_coverage` 记录它对当前位移能量的覆盖率，`fixed_u10_overlap` 记录它与当前前 128 个方向的重合度。若没有可测位移，RMS 和特征值照常记录，方向和累计覆盖率数组为空；参考基不可用时 `fixed_u10_available` 为假。观测步的 `post_update_XXXX.json` 也记录 RQ2 标量及 `timing/rq2_motion_seconds`。早停按原配置每 5 步检查一次。

同一目录下的 `gradient_XXX.npz` 在相同 23 个观测步记录该次 FULL 更新的 hidden 梯度。`optimizer_update` 是这次更新的编号，`checkpoint_before` 是计算梯度时已完成的更新数。令 G 的每一行是当前训练 batch 中一个有效 token 的 KL 损失对最后一层归一化输出的梯度，N 为跨卡汇总的 `valid_tokens`。`eigenvalues` 是未中心化二阶矩 GᵀG/N 的完整降序谱，`own_top128_coverage` 是前 128 项占全谱之和的比例。

从 `gradient_015.npz` 起，`fixed_u10_coverage` 给出 ‖G U₁₀‖²_F / ‖G‖²_F。U₁₀ 取自同一训练轨迹的 `step_010.npz` 的 `basis`，在第 10 次更新完成后产生。后续更新和断点续跑都沿用这组方向。`fixed_u10_available` 标记参考基是否可用。梯度总能量为零时保留全零谱，省略无定义的占比。非零梯度的占比也会写入训练指标 `rq2/gradient_own_top128_coverage` 和可用时的 `rq2/gradient_fixed_u10_coverage`。

RQ3 的配对验证结果在 `training/training_records/rq3/checkpoint_XXX.json`。在默认 106 步 FULL 训练中，0、10、…、100 是更新前的检查点，分别对应第 1、11、…、101 次实际更新。每个文件的 `responses` 有 128 行，同一行内的 `kl_before` 和 `kl_after` 使用相同的生成回答和教师目标，`gain = kl_before - kl_after`。`positive_fraction` 是 `gain > gain_epsilon` 的回答比例。`source_index` 标识验证题，`response_row` 可定位同一检查点 `validation128/checkpoint_XXX.json` 中的回答。`response_sha256` 对有效回答 token ID 按小端 int64 顺序求 SHA-256。第 106 步的终点评估保存在 `validation128/checkpoint_106.json`。

图 7b、7c 在第 1、5、10、…、105、106 次更新及原有详细诊断点（更新前 checkpoint 10、30、60、100，对应更新 11、31、61、101）测量，共 27 个观测点，按实际参数更新编号写入 `training/training_records/rq3/subset_alignment_XXX.json`。例如 `subset_alignment_011.json` 测量第 10 步模型在第 11 次更新前的梯度。按实际参数更新编号调度，复用组同一轮的两次更新分别判断是否观测；`rollout_round` 和 `inner`（从 1 开始）标出对应位置。图 5、6、7a 共用原有的每 10 步验证安排，图 5 另有终点读数。

每次测量用该步的种子把已有的 128 条训练回答分为 4 组，每组 32 条。`subset_train_rows` 给出各组的 batch 行号。`pairs` 保存六对不同子集的全模型参数梯度余弦、完整 hidden 空间内的平均 token 梯度余弦，以及中心化 hidden 梯度协方差前 4 个方向的重合度 ‖VₐᵀVᵦ‖²_F / 4。`means` 和 `valid_pairs` 分别给出三项指标的均值和有效对子数。零范数的余弦、有效秩不足 4 的重合度记为 `null`，计算均值时跳过。各观测点的 JSON 自动进入结果 ZIP。较大的训练子集矩阵只在更新前检查点 10、30、60、100 保存，这四处直接复用详细诊断的计算结果。

流程：先顺序准备共享数据、benchmark和模型，避免并行写入；DeepSeek1.3B、Qwen1.7B、Qwen4B、DeepSeek6.7B按独立单卡槽运行，最多四个并行。全部单卡GPU阶段完成后，Qwen8B→32B、Gemma12B、DeepSeek33B依次独占四卡。Code生成完全部benchmark便释放GPU，将判题提交到单独CPU队列，与后续训练/生成重叠；最多同时判一个实验，全部判题完成后整套才成功结束。Math保留现有Mean@8生成和评分流程。

`--max-parallel-small 1|2|3|4` 控制并行上限；CPU或主存不足时自动减少同时运行的实验数，统计不缩水。各槽使用独立GPU、CPU affinity、Ray本地集群和短临时路径。预留1–8个CPU线程用于判题，其余按槽分配；训练至少保留每个GPU worker一个Ray CPU，另加TaskRunner的一个。只有6个CPU线程时，四卡阶段分配5个给训练、1个给判题，避免等待无法满足的资源请求。BLAS线程数设为1，避免嵌套线程池。

`--gradient-cache-total-gib` 默认64，`--ray-object-store-total-gib` 默认16，分别还受可用主存1/8、1/16限制。单卡并行时按槽平分，四卡阶段独享；原有梯度缓存仍执行自己的内存保护。另按每个单卡任务80 GiB起始预留估计限制并行度。它们是缓冲预算和准入估计，**不是模型、优化器、checkpoint合并与判题的全过程硬内存上限**；33B尤其需要充分主存和磁盘。

正式执行时先按CUDA实际设备顺序解析完整UUID，再按UUID加锁；GPU序号、完整UUID和可识别的UUID前缀共用同一把锁。不要再手动启动占同一组卡的独立任务。失败或中断只终止本套已启动进程组：即使顶层进程已经退出，也继续检查组内残留进程，宽限期后发送SIGKILL；不执行全局 `ray stop`，不删除结果。总调度不自动续训或覆盖已有目录，已完成数据保留；可用下面的单模型入口重试评测。

## 按模型的吞吐起始配置

| 学生 | 卡数 | 任务 | 学生micro-batch | 教师micro-batch | rollout并发 | optimizer/teacher卸载 |
|---|---:|---|---:|---:|---:|---|
| DeepSeek1.3B | 1 | Code | 2 | 2 | 64 | 关/关 |
| Qwen1.7B | 1 | Math | 2 | 2 | 64 | 关/关 |
| Qwen4B | 1 | Math | 2 | 2 | 64 | 关/关 |
| DeepSeek6.7B | 1 | Code | 1 | 1 | 32 | 开/开 |
| Qwen8B→32B | 4 | Math | 2 | 2 | 32 | 开/开 |
| Gemma12B | 4 | Math | 2 | 2 | 32 | 开/开 |
| DeepSeek33B | 4 | Code | 1 | 1 | 16 | 开/开 |

默认 `--profile throughput` 通过现有设置控制批处理与卸载，梯度检查点、rollout显存比例0.30及评测并发64保留。小模型关闭阶段卸载以减少搬运，大模型保留。`--profile conservative` 恢复学生/教师micro-batch=1、并发16及两项卸载。没有启用量化或新的编译路径。**尚无真实H200显存/速度实测，不承诺最优吞吐；批处理及生成调度变化可能改变浮点规约和采样轨迹。**

原来的Top16教师绕过了FSDP逐层CPU卸载，旧`param_offload`字段本身未执行该路径的搬运。本次显式`--teacher-offload on`启用独立的查询阶段卸载：初始化后移至CPU，每次查询前加载，查询结束/异常时先恢复分片再释放。`off`保持教师常驻；没有显式使用新选项的旧入口保持原行为。H200保守预设恢复原配置数值，并让其教师卸载设置实际生效。

## 七个独立入口

```bash
bash scripts/h200/train_eval_deepseek_coder_1_3b.sh
bash scripts/h200/train_eval_deepseek_coder_6_7b.sh
bash scripts/h200/train_eval_deepseek_coder_33b.sh
bash scripts/h200/train_eval_qwen3_1_7b.sh
bash scripts/h200/train_eval_qwen3_4b.sh
bash scripts/h200/train_eval_qwen3_8b.sh
bash scripts/h200/train_eval_gemma3_12b.sh
```

单组入口默认训练一个 FULL 后评测。DeepSeek 默认 Code pass@8（LiveCodeBench、HumanEval+），每题生成 8 个回答，结果写入 `eval/code_pass8/`。Qwen/Gemma 默认数学六项 Mean@8。训练与评测 GPU 相同，单卡 TP=1、四卡 TP=4。`--devices` 指定 GPU，`--task` 切换任务。单独运行一组 RQ4 可用下文的 `--rq4-strategy`。

```bash
bash scripts/h200/train_eval_qwen3_4b.sh --devices 2 \
  --micro-batch-size 2 --teacher-micro-batch-size 2 \
  --rollout-max-num-seqs 64 --output runs/h200/qwen4-custom

# Code训练、合并、全部生成；完成后GPU阶段结束。
bash scripts/h200/train_eval_deepseek_coder_33b.sh --stage train-generate \
  --devices 0,1,2,3 --output runs/h200/deepseek33-run1

# 同一目录，仅CPU判题；不需要可见GPU，也不需要--devices。
CUDA_VISIBLE_DEVICES="" bash scripts/h200/train_eval_deepseek_coder_33b.sh \
  --stage grade --output runs/h200/deepseek33-run1
```

原有 `--stage train`、`eval`保留；`generate`对已有最终checkpoint合并并生成，`grade`仅限Code，跳过训练配置解析和GPU选择/检查，忽略`--devices`。Code的`all`/`eval`先完成全部生成再判题。更换采样/题集/模型需新目录；最终checkpoint完整性检查、合并标记、评测缓存校验保留。

可用 `--student`、`--teacher`、`--cache-dir`、`--model-dir`、`--data`、`--val-data`指定资源。单模型还支持 `--optimizer-offload on|off`、`--teacher-offload on|off`、`--ray-num-cpus`、`--cpu-ids`、`--ray-object-store-gib`、`--gradient-cache-gib`；两个内存上限在独立入口中是该实验预算。

正式启动时自动打印各模型的 GPU 编号与运行顺序，无需单独查看计划。总调度保存 `plan.json`、逐模型状态 `status.json`、`logs/`和各模型目录。每组评测成功后默认清理该组的 `training/checkpoints/` 与 `merged/`；保留训练统计、回答、评测结果和日志。

## 实验数据 ZIP

默认自动打包，**只收集实验数据，不包含任何模型权重或 checkpoint**。评测汇总、生成回答与评分、训练/验证记录、诊断 NPZ/NPY、运行配置和日志会保留；模型目录、私有凭据、链接和嵌套 ZIP 会排除。原始结果不删除。

整套 `--output runs/h200-suite/run1` 完成后生成 `runs/h200-suite/run1-results.zip`；单组 Qwen 8B 默认生成 `runs/h200/8b/8b-math-paired-full-results.zip`。终端打印 `[DOWNLOAD ZIP]` 路径，在服务器文件浏览器中选下载，或用本地 `scp` 获取。下载方法见[快速开始](../../README.md#4-看进度下载结果)。

总调度只在全部 GPU 阶段和 CPU 判题完成后打一次总包，并先释放 GPU 锁；独立脚本每个成功阶段打包，清单会标记阶段。已有实验或中途结果可手动打包，完全不需要加载模型：

```bash
python scripts/package_results.py --run-dir runs/h200-suite/run1
```

`--no-package` 关闭自动打包。ZIP64 支持大文件，包内 `EXPORT_MANIFEST.json` 提供文件清单及 CRC32；重复运行会更新本实验的包，写入失败保留此前完整 ZIP。

查看全部参数：

```bash
python scripts/run_h200_suite.py --help
python scripts/run_h200_experiment.py --help
```

## 重试最终评测

已有完整训练 checkpoint 时，使用原输出目录：

```bash
bash scripts/h200/train_eval_qwen3_4b.sh \
  --stage eval --devices 0 --output runs/h200-suite/run1/4b
```

此命令适用于评测失败后保留的 checkpoint，或启动时设置了 `--keep-model-files` 的实验。评测成功并自动清理权重后，不能再通过 `--stage eval` 重新生成回答；已保存的回答和统计仍保留。合并中断且没有完成标记时，先将该实验的 `merged/` 移到另一个位置后重试。训练中断需使用新输出目录重新训练。


## 单独补测任一未训练 Base

在含 `train.py` 的项目目录运行。每组使用 `configs/<规模>/models.json` 固定的原始学生权重。下面分别以论文表 3 的 Qwen3 4B 和四卡的 DeepSeek-Coder 33B 为例。代码评测使用训练后评测的同一套题库、提示模板和判题环境。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_base.py --scale 4b
python scripts/prepare_code_benchmarks.py --output-dir data/code_benchmarks
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/evaluate_base.py --scale deepseek-coder-33b
```

默认输出目录为 `runs/h200/<规模>/math-base/` 或 `runs/h200/<规模>/code-base/`，可用 `--output` 更改。数学组评测的 MATH500、HMMT25、HMMT26、AIME26、Minerva 分别有 500、30、33、30、272 题。每题以 temperature=1、top_p=1、seed=42 生成 8 个回答。先算各题集的正确回答比例，再对五项等权平均。代码组在 LiveCodeBench 和 HumanEval+ 每题生成八份候选，保存各题集的 pass@8 及两项等权平均。

数学 Base 在结果目录保存以下文件。

| 文件 | 内容 |
| --- | --- |
| `responses.jsonl.gz` | 每题一行。保存题目、标准答案、实际提示词、8 个回答全文、长度和停止原因。直接写入 gzip 临时文件，完成后在同一目录改名。 |
| `responses.meta.json` | 固定模型版本、模型文件和题集哈希、采样配置、提示词编码版本、压缩原文的 SHA-256。 |
| `scores.jsonl` | 每完成一道题追加一行。用 `index`（从 0 开始）关联原文，以 `generation_sha256` 核对来源。`row.acc_list` 和 `row.strict_last_box_acc_list` 分别是这 8 个回答的宽松和严格判分，顺序和原文相同。多进程判分的写入顺序可以与题序不同。 |
| `summary.json` | 五项成绩、等权平均、题数、样本数以及原文和判分文件的实际字节数。表 3 取 `lenient_answers_v1.mean_at_8` 和 `lenient_answers_v1.five_benchmark_mean_at_8`，乘 100 得到百分数。 |

原文整批写完并校验后，同目录重试可直接接着判题。生成或压缩中断且原文尚未写完时，需要重新生成。`.base-evaluation.lock` 防止两个评测同时写同一目录。旧格式的完整生成缓存通过校验后原地复用，已有的旧文件保留。模型权重位于共享 Hugging Face 缓存，可用 `--cache-dir` 指定已有缓存。

独立评测成功后，默认在其他 4B 实验 ZIP 所在目录生成 `runs/h200/4b/4b-math-base-results.zip`。自定义 `--output` 时，ZIP 保存在该输出目录旁。终端的 `[DOWNLOAD ZIP]` 会打印实际路径。`--no-package` 可关闭打包。

默认 H200 套件对全部七个原始学生模型各测一次 Base，每组在 `<规模>/eval/base/` 保存，和训练结果进入同一 ZIP。四组数学沿用表 3 五项 Mean@8，RQ4 采用严格判分；三组 DeepSeek 用相同的 chat 提示和双 EOS，评测 LiveCodeBench、HumanEval+ 的 pass@8，GPU 生成后由 CPU 判题。每组四行 Base、FULL、早停、复用，共 28 行结果、14 次训练。

单独补测用 `python scripts/evaluate_base.py --scale <规模>`，可用 `--output` 指定目标，默认单独打 ZIP。代码任务使用与训练后评测相同的已准备 benchmark 目录（可用 `python scripts/prepare_code_benchmarks.py` 准备），支持 `--generate-only` 和 `--grade-only`。单独运行时请设置与该组一致的可见 GPU。原有 4B 脚本继续可用。

## RQ4 策略对照

总入口为每组所选模型在相同 GPU 配置上依次运行 FULL 和复用。早停成绩来自同一条 FULL 轨迹。运行 `bash scripts/h200/run_all.sh --scales 1.7b 4b --output runs/h200-suite/rq4` 会执行四次训练，产出两个规模各三行策略结果。单模型默认入口和显式 `--rq4-strategy full` 同样会评测早停点；显式 `--rq4-strategy early-stop` 仍可单独训练至早停。

* **早停**。每 5 次参数更新在固定 probe 上检查一次。从第 15 步起，窗口覆盖 t−10、t−5、t。`training/training_records/rq4/early_stop_XXX.json` 保存判定依据。首次触发时，FULL 把当前权重保存到 `training/checkpoints/early_stop/actor/`，然后继续训练。训练结束后对这份权重运行与 FULL 相同的评测。数学回答保存在 `eval/early-stop/final_mean8.jsonl`，代码回答及判题结果按题集保存在 `eval/early-stop/code_pass8/`。代码生成结束后释放临时合并模型，由 CPU 判题。到预算终点才选中最终模型时，共用 FULL 的权重和逐题结果。`training/training_records/rq4/early_stop_summary.json` 记录来源、实际步数、判定原因、回答路径和累计训练耗时，随结果 ZIP 导出。
* **复用**。每次新 rollout 和教师目标供两次真实 AdamW 更新使用，53 次 rollout 共 106 次更新。在 rollout 第 5、10、…、50、53 轮，`reuse_round_XXX.json` 记录同一训练 batch 按有效 token 平均的 `kl_before_first`、`kl_after_first` 和 `kl_after_second`。最后一项用第二次更新后的学生重新前向计算。`second_gain = kl_after_first - kl_after_second`，正值表示第二步进一步降低了 KL。
* **时间与评测**。每次独立训练的 `rq4_training_time.json` 记录训练命令的实测墙钟时间，包含训练初始化、诊断和保存 checkpoint。共享早停行以 `source_strategy=full` 标记，计时截止于早停权重保存完成；共用预算终点时采用 FULL 总时长。合并及 benchmark 评测单独执行。续跑累计有记录的训练尝试，无法还原完整时长时汇总值为 `null`。
* **对照表**。套件 ZIP 的 `rq4_comparison.json` 按规模和策略汇总上述结果。代码组保存两项 pass@8、等权平均和相对 FULL 的百分点变化 `two_benchmark_change_pp`。数学五项为 MATH500、HMMT25、HMMT26、AIME26 和 Minerva，各项 Mean@8 和算术平均用 0–1 表示，`five_benchmark_change_pp` 给出相对同规模 FULL 的百分点变化，`training_time_saved_fraction` 给出实测时间节省比例。含 AMC23 的数学原始六项评测在相应训练目录的 `eval/final_mean8.summary.json`，提前触发的早停点在 FULL 的 `eval/early-stop/final_mean8.summary.json`。

## 磁盘与自动清理

七组 paired-FULL 均在第106步保存最终模型，常规终点保留1个 checkpoint，提前触发时另保留1个早停 checkpoint，保存内容仅为模型，不保存优化器状态。

默认每组完成全部评测后，立即删除该组输出目录内的 `training/checkpoints/` 和 `merged/`，无需等七组全部结束。Code 先完成两项 benchmark 的 CPU 判题再清理；只训练或只生成回答的阶段保留权重，评测失败也保留权重以便重试。

- 保留 `training/replication/`、`training/training_records/`、`eval/`、`logs/` 及其中的统计、表征分析、回答和评分。
- `model_cleanup.json` 记录删除路径、文件大小与完成状态；`model_artifact_metadata.json` 保留 checkpoint 的配置和步数等文本元数据。两份记录进入结果 ZIP。
- 不删除共享 Hugging Face 缓存、原始学生/教师模型、数据集或其他实验目录。
- 需要保留训练后权重时，在总入口或单模型入口添加 `--keep-model-files`。默认不需要额外参数。
- 单模型入口显式指定 `--model-artifacts-root` 时，checkpoint 和合并权重写到外部目录。默认在完整评测成功后清理本次运行拥有的权重，`--keep-model-files` 可保留它们。显式指定 `--save-freq` 会生成可续训的 checkpoint，同时保存模型、优化器和额外状态，磁盘预算需要相应增加。

默认磁盘余量为 5 GiB，可通过环境变量 `OPD_MIN_FREE_GIB` 调高。Base 首次下载前按固定模型版本的实际缺失文件大小检查缓存盘，在写压缩原文、判分和汇总前检查输出盘。打包时按实际写出的 ZIP 字节检查目标盘，旧 ZIP 保留到新 ZIP 成功完成；打包失败时移除本次临时 ZIP。总套件启动各子任务前及运行中约每秒检查输出、数据、临时目录和 Hugging Face 缓存所在的文件系统，低于余量会中止套件。监控依据文件系统可用空间。独立用户配额、其他并发写入和进程退出延迟仍会影响实际余量。

运行整套时，模型合并阶段的峰值包含训练 checkpoint 和合并后的权重，且套件会预先缓存所选模型。任一规模提前触发早停时，还会多一份只含模型权重和配置的 checkpoint。正常流程先合并并评测早停模型，释放临时合并副本后再合并 FULL。成功清理时打印本组 checkpoint 和合并权重的逻辑字节数。共享下载缓存跨实验保留。打整套 ZIP 时，源结果和 ZIP 同时占用空间；`--no-package` 可跳过打包。

按FP32 checkpoint加BF16合并模型估算，两份临时模型约为每参数6字节；33B约198 GB（184 GiB）；提前触发时再计一份按 FP32 估算约 132 GB 的早停分片，合计约 330 GB（307 GiB），共享缓存及统计另计。33B 的 FULL 判题完成后才启动同规模复用训练。实际大小以本次文件记录为准，原始模型下载缓存、数据集、统计与ZIP占用另算。默认逐组清理能避免七组完成后的权重副本长期累积，但不能消除当前正在训练/合并/评测那一组的磁盘峰值。

统计协议和保存频率保持原样。逐token参数梯度诊断保存能量及抽样统计，不把每个token的完整参数梯度向量写入磁盘。


## 轻量保存与新增统计

默认保留原统计频率和参数梯度抽样数。对称矩阵无损保存上三角，结果 ZIP 自带 `READ_MOMENTS.py` 读取工具；不再默认保存完整 hidden 梯度样本向量。新增 step-10 之后及相邻步的表征变化统计，以及已有验证诊断时点的参数变化比例、大小和层分布，不新增模型前向、反向或 SVD。验证结果固定写入训练输出目录，外置权重目录也会在完整评测成功后按归属记录清理。已有未知外置目录不会自动删除。

保存字段、断点恢复边界及读取方法见 [诊断保存说明](../../docs/diagnostic_storage.md)。新增 CPU 汇总耗时单独记录，不承诺零运行开销。

## 重型诊断频率

训练 KL、固定高 KL 阈值占比和整批 hidden 梯度能量占比继续逐次更新记录。逐 token 全模型参数梯度、固定 probe 表征及完整 hidden 梯度谱在更新 1、5、10、…、105、106 测量（23 点）；每点仍抽样 256 个 token，抽样、权重、梯度定义不变。未测量步的 RQ1 文件标记 `parameter_gradient_measurement.measured=false`，参数能量项留空，不插值、不沿用旧值。全量模式也沿用诊断步调度。

跨 batch 梯度在上述 23 点及原有 4 个验证配对诊断点测量（共 27 点）。验证集的每 10 步安排、首尾评测和 early-stop 的每 5 步检查保持原样。Reuse 使用实际 optimizer update 编号调度。

表征增量的 `previous_observation_*` 表示相对上一个实际观测点的变化，`previous_observation_checkpoint` 和 `previous_observation_update_gap` 标明区间；`previous_step_*` 仅在相邻更新都有观测时存在。恢复后若没有上一个观测的 hidden，则相应可用标记为 false。

降低观测频率减少时间采样密度，保留点的测量算法不变；不等于总训练速度提高 4.61 倍。正在运行的 Python 进程不会因拉取新提交而自动改变频率。

### 统计计算与半小时预算

逐 token 参数梯度能量继续在 GPU 上归约，保留合并输出头查询与 Triton FP64 融合归约。CPU 或无法使用融合路径时，回退实现默认每块 4,194,304 个元素，使用一个约 32 MiB 的 FP64 临时缓冲，避免数千次小块转换及额外平方缓冲；共享权重、FSDP padding 和参数覆盖范围保持原定义。分块求和顺序变化可能产生 FP64 末位差异。

梯度缓存充分时，四组梯度的十个 Gram 条目在一次分块遍历中计算，共享 FP64 转换并合并进程间汇总；不增加整模型梯度缓存预算。缓存不足时继续使用原有的有界重算路径。

跨 batch 的 CPU 谱分解使用单个后台线程，与后续组的 GPU 梯度计算重叠。后台只读取已复制到 CPU 的矩阵；模型、优化器、CUDA 和进程间通信仍由主线程管理。报告返回前等待后台结果，异常通过原有机制通知全部进程；不丢弃观测或跳过失败的统计。

日志新增 `[parameter tokens] stage seconds`，分别报告 `forward`、`backward`、`parameter_energy`。原始记录的 `parameter_gradient_measurement.stage_timing` 保存相同数据，训练指标使用 `timing/parameter_token_<stage>_seconds`。CUDA 使用事件计时，不在 token 循环中强制同步；`host_stage_seconds` 是另一套主机提交耗时，不能与设备时间相加。模型加载、验证、CPU 谱分解等不包含在这些 token 阶段中。

额外统计不超过 30 分钟是待 H200 计时验证的目标，不是当前代码已证明达到的性能。默认 23 个重型观测点对应全部统计平均每点约 78 秒，另有验证和配对诊断成本。实现不会为了满足预算而静默降低样本数、跳过观测，或用 hidden 梯度替代参数梯度。
