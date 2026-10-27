# Phase41：Phase39 BASE/FUSION 补齐五折

本批只训练 8 项：`P41_BASE_F02`～`F05` 和 `P41_FUSION_F02`～`F05`。`P39_BASE_F1`、`P39_FUSION_F1` 必须按 `configs/phase41_plan.yaml` 中的 SHA-256 复用，脚本不会重训 Fold1。所有任务保持 Phase39 Fold1 的 384 输入、ImageNet 初始化、BCE、AdamW、EMA=0.999、最多 30 Epoch、patience=6、Study 级 OOF Macro-AUC 早停；BASE/FUSION 都固定 micro-batch 4、累积 4、DataLoader worker 0，使有效 batch 16 和 BatchNorm 微批条件相同。

## 朋友的一次准备

从私有仓库克隆 `friend-gpu-batch`，检出交付消息给出的完整 commit。使用独立克隆目录，不更新正在运行的其他队列。沿用现有 Linux PyTorch 环境；依赖见 `requirements-phase32.txt`。把赛事授权的 `cxr_first.zip`、合法 ImageNet `hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth`、冻结三份 Fold CSV，以及本地 Phase39 两个 Fold1 的 `ema_best.pth`、`oof_predictions.csv`、`resolved_config.json`、`run_manifest.json` 放到服务器授权目录。Fold1 源目录应为 `fold1_results_root/P39_BASE_F1/` 和 `fold1_results_root/P39_FUSION_F1/`；这些权重不在普通 Git 仓库，缺失或哈希不符时预检拒绝启动。

复制 `configs/phase41_server.example.yaml` 为 **不提交 Git 的** `configs/phase41_server.local.yaml`，填写五个绝对路径、授权的 `gpu_ids`、`max_concurrent`、`gpu_hours`、最低空闲磁盘/RAM。`gpu_hours` 不得留空。8 项每项硬超时 48 GPU 小时，调度器在派发前保守预留超时额度；预算较小时任务留在 PENDING，不超支。预检也计入预算。8 卡全部同时派发需至少 384 GPU 小时加预检预留；这只是安全上界，**不是** 4090 实测工时估计。需要先确认其他队列已退出，统一主机锁禁止两套队列同时接管 GPU。

默认 `upload.enabled: false`，结果留在本机。若朋友已同意自动回传并具备私有仓库写权限、`gh auth login` 和 Git 认证，可改为 `true`。此时轻量 CSV/JSON/YAML/日志增量推到独立分支 `results/phase41-fivefold` 的 `results/phase41_batch/`，最佳 EMA 推理权重按单文件传到私有 Release `phase41-fivefold-v1`。不上传原始图像、原始标签或恢复训练状态；不覆盖历史结果。

## 唯一训练启动命令

```bash
python scripts/phase41_run.py --config configs/phase41_server.local.yaml
```

该命令先做 Linux 资源、8 张 4090、空闲 GPU、合法预训练 SHA、赛事数据、患者划分、Fold1 来源、各授权 GPU 的真实 AMP 优化器步进、CUDA checkpoint 恢复、Val/Test 短推理检查。成功后才派发 8 项任务，每个 GPU 同时最多一项。真实多卡长时吞吐、磁盘竞争和 4090 性能仍只能在朋友服务器观察；预检失败时不启动正式任务。

```bash
python scripts/phase41_run.py --config configs/phase41_server.local.yaml --action status
python scripts/phase41_run.py --config configs/phase41_server.local.yaml --action stop
python scripts/phase41_run.py --config configs/phase41_server.local.yaml --action resume
```

`stop` 只阻止新任务，已启动任务自然完成；再次运行唯一启动命令会复用有效预检和 DONE，并从完整 Epoch checkpoint 恢复中断任务。显式 STOP 后用 `resume`。不要在尚有孤儿训练进程时重启队列。网络回传失败不会删除模型；修复网络后用 `--action upload` 重试，不重训。

## 结果位置与判读

`output_root/experiments/P41_*` 每项立即保存 `ema_best.pth`、`raw_at_ema_best.pth`、`last_training.pt`、`resolved_config.yaml`、`train_history.csv`、`train.log`、OOF/Val/Test Study 概率、`metrics.json`、逐类指标、manifest、大小与 SHA-256 收据 `DONE.json`。未完成项保留失败状态和 checkpoint。`output_root/reused_fold1/` 是哈希固定的 Fold1 拷贝及补充 Val/Test 推理，原始 Fold1 不被修改。

`reports/experiment_status.csv` 显示进度。完成全部五折后，`oof/{BASE,FUSION}_oof_all.csv`、`validation/{BASE,FUSION}_val_fivefold.csv`、`reports/fivefold_summary.json` 给出两个骨干的完整 OOF、五折 Val、BASE/FUSION 的完整及排除 Fold1 患者级配对 Bootstrap；缺折时只写 `PARTIAL_OOF`，不冒充完整五折。官方 Val 已被重复用于开发，不能视为独立测试；Test 无标签，不计算 Test AUC。程序不生成 Submission，也不上传比赛平台。

本地 CPU 测试证据见 [验收记录](acceptance.md)。真实 Linux 8×4090、共享磁盘与完整训练仍为远程待验证，不能据本地测试声称通过。单个模型的大小报告仅是实测推理文件大小；无目标 T4 端到端实测时一律 `T4_UNVERIFIED`。完整十模型集成也不能默认满足 500 MB 或 100 ms。
