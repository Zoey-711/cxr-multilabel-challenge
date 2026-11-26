# Phase17-Batch：给朋友的一次部署说明

**当前仅代码准备，尚未授权在朋友集群正式启动。** 本地Phase17另行运行，不能复制其活动目录作为本队列输出。所有任务使用独立的 `$CXR_OUTPUT_ROOT/phase17_batch/`；当输出基目录为项目的 `outputs` 时，目录就是 `outputs/phase17_batch/`。

## 朋友只需三步

1. 将私有代码包、赛事允许传输的原始ZIP、三份合法ImageNet权重放到授权目录。不要上传比赛数据或标签到公开GitHub。原ZIP只读、无需解压、各折共享。
2. 填写下列环境变量，并经双方同意GPU数量及总GPU小时。路径示例需要换成朋友实际授权位置。
3. 获得正式启动批准后，只执行下述一条启动命令；安装检查、GPU预检、队列、验证、OOF、报告和打包自动完成。

```bash
export CXR_DATA_ROOT=/authorized/data/cxr_first.zip
export CXR_OUTPUT_ROOT=/authorized/outputs
export CXR_PRETRAINED_ROOT=/authorized/torch_cache
export CXR_BACKUP_ROOT=/authorized/backup
export CXR_GPUS=0
export CXR_GPU_HOURS=80
bash scripts/friend_start.sh --config configs/batch_plan.yaml
```

`80` 是示例预算，必须替换为朋友批准的额度。每GPU同时最多一个模型，不把单模型自动改成DDP。预算按占用GPU的任务墙钟小时累计；下一任务须能预留其整个timeout才启动，因此可能提前停止、留下未完成任务。正在运行任务有超时上限。不会自动扩展预算。

## 环境与数据

- Python3.12、torch2.10.0、torchvision0.25.0、CUDA12.8；不升级GPU驱动。不兼容时停止并报错。
- 默认复用镜像框架。确需安装官方cu128 wheel时，由朋友显式 `export CXR_INSTALL_FRAMEWORK=1`；安装脚本不自动作此决定。
- 预训练目录必须是 `CXR_PRETRAINED_ROOT/hub/checkpoints/{densenet121-a639ec97.pth,convnext_tiny-983f1562.pth,swin_t-704ceda3.pth}`。
- 原ZIP SHA-256：`ff458640e9974d63aa3329ee7ae1248260b619223980218e3741aee4c6d6836f`。首次完整校验约55.23GB读盘；不重复上传/解压。
- 数据约55.23GB；输出预检需至少20GB空闲，备份位置建议另留20GB。结果分卷打包有临时空间成本；共享网络盘的ZIP随机读取需实测。
- 三个冻结fold索引和固定标签在交付代码包内。禁止重新生成患者划分。官方Val只选Epoch；OOF标签不参与训练或早停。

## 计划与预先固定的差异

基础档：`D17_F01..F05`、`C17_F01..F05`、`S17_F01..F05`，共15次。分别以B01/M02/T01实际配置为控制，effective batch16，patience8/6/6，最多30Epoch，384保比例Padding、原增强和BCE不变。

EMA：仅C17在同次训练维护，decay=.999；第一个成功optimizer step直接复制，此后每个成功step更新；AMP跳步不更新。ConvNeXt无BatchNorm，保留buffer同步语义，不做BN重估。原始模型负责早停，普通与EMA各自保存最佳Val；不根据中途结果改EMA。额外成本为EMA参数更新、GPU约一份模型、每轮一次Val和最后额外Val/OOF推理。

C组：D17_SEED43和C17_SEED43两个完整Train模型预备，默认关闭，仅A组完成、资源足够且另行批准后使用 `--allow-extended`。D组：最多两个占位，不包含任何未批准架构或可启动任务。不会因资源剩余自动新增模型。

固定比较：E07、D5、C5、S5、E17=.25D5+.25C5+.5S5；EMA只做预定C分支替代。部署小候选固定F01三骨干，以及D5+C17_F01+S17_F01，不根据单折成绩挑选“最佳折”。完整15模型预估约1,251,238,365bytes，超过500MB；D5+C1+S1约364,102,553bytes。最终用实际导出纯state_dict字节数复核。

## 耗时与预算（不是A100实测）

历史4060记录：B01平均7.35分钟/Epoch（13轮），M02为22.97分钟（11轮），T01为22.57分钟（13轮）。按4/5训练量粗估A组分别6.37、16.84、19.56GPU小时，合计42.78；全部30轮约105.78GPU小时。此缩放低估固定Val/启动成本，另预留15%–25%以及EMA、独立Val/OOF、备份时间。不会保证复现历史早停轮数。

| 场景 | A组按历史轮数的示例墙钟 | 全30轮示例 |
|---|---:|---:|
| 1张A100，假设整体速度为4060的2–4倍 | 10.7–21.4小时 | 26.4–52.9小时 |
| 3张A100，假设同上且充分并行 | 理想3.6–7.1小时，实际受分支长短/I/O限制 | 理想8.8–17.6小时 |

以上只是假设情景，不是A100实测；小micro-batch未必充分利用A100。B组开销必须由实测Val吞吐计算，先预留A组的5%–15%，不能当作测量值。C组本地参考另约5.80GPU小时（全30轮15.16），实际加评估。D组当前0任务、0GPU小时。

自动预检每GPU每骨干2个真实更新步和短测速；`batch_benchmark.py`分别估计ZIP读取与缓存GPU计算。它不是完整训练计时；结果报告包含该限制。以实测速率重新估算，不按本地速度冒充A100性能。具体原始历史数字见 `docs/batch17_costs.json`。

## 运行、停止、恢复与Slurm

建议在朋友已有tmux中执行唯一启动命令，SSH断线不影响训练：

```bash
tmux new -s phase17
bash scripts/friend_start.sh --config configs/batch_plan.yaml
# Ctrl-b d 分离；tmux attach -t phase17 恢复查看
source .venv/bin/activate
python scripts/batch_run.py --config configs/batch_plan.yaml --action status --root "$CXR_OUTPUT_ROOT/phase17_batch"
python scripts/batch_run.py --config configs/batch_plan.yaml --action stop --root "$CXR_OUTPUT_ROOT/phase17_batch"
```

stop为**安全排空**：当前任务完成后不再启动新任务，不会即时杀训练；超时只终止队列自己启动的进程树。恢复前确认原队列已退出，删除该目录的STOP标记，再运行同一启动命令。恢复从最近完整Epoch，不承诺恢复Epoch中间尚未保存的batch。优化器、调度器、scaler、RNG、早停、EMA和最佳产物一起恢复；已完成模型不重训。OOM/数值异常/哈希不符不自动改参数。

```bash
rm -- "$CXR_OUTPUT_ROOT/phase17_batch/STOP"
bash scripts/friend_start.sh --config configs/batch_plan.yaml
```

崩溃遗留queue.lock时，核实其中host/PID已经退出后才移走锁；不能对活跃队列解锁。新环境恢复还需重新预检。故障重试上限1次，仅明确的临时I/O等可恢复错误，配置不变。

Slurm使用朋友已有调度系统和授权资源，不自行安装：

```bash
sbatch --partition=<授权分区> --account=<授权账号> --gres=gpu:<授权数量> --cpus-per-task=<授权CPU数> --mem=<授权内存> --time=<授权时限> scripts/friend_slurm.sh --config configs/batch_plan.yaml
```

保留Slurm的CUDA_VISIBLE_DEVICES，禁止选择分配范围外GPU。此版本为单节点多GPU队列；多节点需各自独立输出/任务分配，不可多个调度器共享一个队列目录。

## 自动产物与交付

`phase17_batch/{manifests,experiments,oof,validation,analysis,ensembles,reports,delivery}`。

每模型完成后自动生成要求的 `.pth`纯推理权重、完整配置、训练历史、指标、Val/OOF概率、resume、manifest和DONE；失败有FAILED/error.log。每次任务完成复制到授权备份路径。基础队列停止（包括预算不足或失败）也输出明确缺失表和已完成结果包。

最终只需提供 `delivery/latest.json` 指向的目录下载路径，其中有results、models分卷、manifest、README_results和COMPLETE通知。不会自动发送消息、上传Github、删除远端文件或生成排行榜提交。

可选手动重新打包（新增版本，不覆盖已有包）：

```bash
python scripts/batch_delivery.py --config configs/batch_plan.yaml --action pack --root "$CXR_OUTPUT_ROOT/phase17_batch" --include-resume
```

故障处理：环境错误→检查版本；OOM→任务停止，人工批准新版本配置后才能重跑；哈希错误→修复数据/权重，不重划分；磁盘满→增加授权空间或下载已校验结果，禁止脚本自动删除冻结模型；部分OOF→显示incomplete。正式部署还需朋友GPU真实预检通过，本地合成验收不能替代。
