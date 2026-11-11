# H200 七组训练与评测

环境、模型权限和数据准备见项目主 [README](../../README.md#h200-操作指南)。从项目根目录运行，使用已激活的 Python 3.10 训练环境。

## 整套实验

```bash
# 只查看计划，不下载、不写运行目录、不占用 GPU。
bash scripts/h200/run_all.sh --devices 0,1,2,3 --dry-run

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

`--max-parallel-small 1|2|3|4` 控制并行上限；CPU或主存不足时自动减少同时运行的实验数，统计不缩水。各槽使用独立GPU、CPU affinity、Ray本地集群和短临时路径。预留2–8个CPU线程用于判题，其余按槽分配；四卡阶段使用其余全部CPU。BLAS线程数设为1，避免嵌套线程池。

`--gradient-cache-total-gib` 默认64，`--ray-object-store-total-gib` 默认16，分别还受可用主存1/8、1/16限制。单卡并行时按槽平分，四卡阶段独享；原有梯度缓存仍执行自己的内存保护。另按每个单卡任务80 GiB起始预留估计限制并行度。它们是缓冲预算和准入估计，**不是模型、优化器、checkpoint合并与判题的全过程硬内存上限**；33B尤其需要充分主存和磁盘。

两个总调度实例通过GPU ID锁避免重复占卡；不要再手动启动占同一组卡的独立任务。失败或中断只终止本套已启动进程组，不执行全局 `ray stop`，不删除结果。总调度不自动续训或覆盖已有目录，已完成数据保留；可用下面的单模型入口重试评测。

## 按模型的吞吐起始配置

| 学生 | 卡数 | 任务 | 学生micro-batch | 教师micro-batch | rollout并发 | optimizer/teacher卸载 |
|---|---:|---|---:|---:|---:|---|
| DeepSeek1.3B | 1 | Code | 4 | 4 | 64 | 关/关 |
| Qwen1.7B | 1 | Math | 4 | 4 | 64 | 关/关 |
| Qwen4B | 1 | Math | 4 | 4 | 64 | 关/关 |
| DeepSeek6.7B | 1 | Code | 2 | 2 | 32 | 开/开 |
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

它们仍可单独使用，默认训练后评测。DeepSeek默认Code pass@4（LiveCodeBench、MBPP+、HumanEval+），Qwen/Gemma默认数学六项Mean@8；训练和评测GPU列表相同，单卡TP=1、四卡TP=4。`--devices`指定所需一张/四张卡，`--task`切换任务。入口不自动跑RQ4 early-stop/reuse或Base基线。

```bash
bash scripts/h200/train_eval_qwen3_4b.sh --devices 2 \
  --micro-batch-size 4 --teacher-micro-batch-size 4 \
  --rollout-max-num-seqs 64 --output runs/h200/qwen4-custom

# Code训练、合并、全部生成；完成后GPU阶段结束。
bash scripts/h200/train_eval_deepseek_coder_33b.sh --stage train-generate \
  --devices 0,1,2,3 --output runs/h200/deepseek33-run1

# 同一目录，仅CPU判题，不加载模型或检查GPU。
bash scripts/h200/train_eval_deepseek_coder_33b.sh --stage grade \
  --devices 0,1,2,3 --output runs/h200/deepseek33-run1
```

原有 `--stage train`、`eval`保留；`generate`对已有最终checkpoint合并并生成，`grade`仅限Code。Code的`all`/`eval`先完成全部生成再判题。更换采样/题集/模型需新目录；最终checkpoint完整性检查、合并标记、评测缓存校验保留。

可用 `--student`、`--teacher`、`--cache-dir`、`--model-dir`、`--data`、`--val-data`指定资源。单模型还支持 `--optimizer-offload on|off`、`--teacher-offload on|off`、`--ray-num-cpus`、`--cpu-ids`、`--ray-object-store-gib`、`--gradient-cache-gib`；两个内存上限在独立入口中是该实验预算。

总调度保存 `plan.json`、逐模型状态 `status.json`、`logs/`和各模型目录；模型目录包含`training/`、`merged/`、`eval/`。查看全部参数：

```bash
python scripts/run_h200_suite.py --help
python scripts/run_h200_experiment.py --help
```
