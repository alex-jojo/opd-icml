# H200 高级运行参考

首次部署和完整七组启动见[快速开始](../../README.md)。以下命令仅在单独运行、调参或恢复评测时使用；均在已激活的 Python 3.10 环境中执行。

所有模型、数据、配置、输出和 ZIP 路径均以项目根目录（含 `train.py` 的目录）为基准；例如 `models/student`、`data/benchmarks.jsonl`、`runs/h200-suite/run1`。入口会自行定位项目根目录，从其他工作目录启动也适用。项目外资源可写为 `../shared/...`，运行计划与新生成的路径标记保存相对路径。

所有入口继承当前终端的 `WANDB_API_KEY`、`WANDB_MODE`、`WANDB_PROJECT` 和可选的 `WANDB_ENTITY`。无需配置文件；设置方法见[快速开始](../../README.md#2-登录模型账号设置-wb)。

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

七组均为 paired-full：106轮、每轮一次更新、全局batch/mini-batch=128、学习率1e-5。参数梯度每个观测点抽样256个token（64 high-KL+192 remaining），每个样本仍求完整可训练参数梯度，hidden统计保留。`--parameter-token-gradients exact-all` 可显式切换全量。总调度不减少样本、频率、回答长度、评测次数或精度。

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

它们仍可单独使用，默认训练后评测。DeepSeek默认Code pass@8（LiveCodeBench、MBPP+、HumanEval+），每题生成8个回答，结果写入 `eval/code_pass8/`；旧 `code_pass4/` 结果保留，不作为 pass@8 缓存复用。Qwen/Gemma默认数学六项Mean@8；训练和评测GPU列表相同，单卡TP=1、四卡TP=4。`--devices`指定所需一张/四张卡，`--task`切换任务。入口不自动跑RQ4 early-stop/reuse或Base基线。

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


## 磁盘与自动清理

七组 paired-FULL 均在第106步保存最终模型，最多保留1个 checkpoint，保存内容仅为模型，不保存优化器状态。

默认每组完成全部评测后，立即删除该组输出目录内的 `training/checkpoints/` 和 `merged/`，无需等七组全部结束。Code 先完成三项 benchmark 的 CPU 判题再清理；只训练或只生成回答的阶段保留权重，评测失败也保留权重以便重试。

- 保留 `training/replication/`、`training/training_records/`、`eval/`、`logs/` 及其中的统计、表征分析、回答和评分。
- `model_cleanup.json` 记录删除路径、文件大小与完成状态；`model_artifact_metadata.json` 保留 checkpoint 的配置和步数等文本元数据。两份记录进入结果 ZIP。
- 不删除共享 Hugging Face 缓存、原始学生/教师模型、数据集或其他实验目录。
- 需要保留训练后权重时，在总入口或单模型入口添加 `--keep-model-files`。默认不需要额外参数。

每次启动显示输出文件系统的剩余空间，清理结束显示删除文件的逻辑大小。该值不是硬链接、压缩或文件系统配额下的实际可用空间增量。

按FP32 checkpoint加BF16合并模型估算，两份临时模型约为每参数6字节；33B约198 GB（184 GiB）。实际大小以本次文件记录为准，原始模型下载缓存、数据集、统计与ZIP占用另算。默认逐组清理能避免七组完成后的权重副本长期累积，但不能消除当前正在训练/合并/评测那一组的磁盘峰值。

统计协议和保存频率保持原样。逐token参数梯度诊断保存能量及抽样统计，不把每个token的完整参数梯度向量写入磁盘。
