# Phase 17-Batch：朋友服务器运行说明

本分支只交付代码和冻结协议。运行目标为**一台 Linux 服务器，最多 8 张 NVIDIA RTX 4090（每张 24GB）**，共享文件系统；每张卡同时最多运行一个独立模型任务。没有 DDP、`torchrun` 或跨节点通信。代码仓库不包含约 55GB 比赛 ZIP、冻结患者索引、标签、E07 参考概率、ImageNet 权重、私钥或令牌。请通过比赛允许的独立渠道取得这些文件。用户或 Codex 不登录朋友服务器。

## 朋友的三步

1. 克隆用户提供的**私有**仓库 `friend-gpu-batch` 分支，并检出用户确认的固定 commit。将官方 `cxr_first.zip`、冻结协议目录和六种合法 torchvision ImageNet 权重放在授权存储位置。协议目录保留仓库相对路径，例如 `batch_assets/train_labels.csv`、`batch_assets/val_labels.csv`、`batch_assets/e07_official_val.csv`、`outputs/analysis/phase17_patient_folds/*.csv`；脚本逐个核对冻结 SHA-256，绝不重划 Fold。
2. 复制 `configs/friend_server.example.env` 为 `configs/friend_server.local.env`，一次填好绝对路径、授权 GPU 编号、最多并发数和**朋友明确批准的最大 GPU 小时**。本地配置已被 `.gitignore` 排除。缺少预算或路径会停止。默认写了 `0,1,2,3,4,5,6,7` 与并发 8；实际主机 CPU、可用内存、磁盘和 GPU 必须通过预检，否则降低授权并发再启动。`CXR_DATA_ROOT` 指向官方 ZIP；`CXR_OUTPUT_ROOT` 是结果所在父目录，最终写入其中的 `phase17_batch/`；`CXR_BACKUP_ROOT` 指向独立授权备份目录。
3. 在仓库根目录执行唯一启动命令：

```bash
bash scripts/friend_start.sh --config configs/friend_server.local.env
```

启动入口自动安装所需 Python 包（优先复用兼容的已有 PyTorch/CUDA 环境，不改驱动）、导入并校验冻结协议、核对数据及预训练权重哈希、做少量真实 GPU 步与吞吐预检，然后调度 22 项基础任务。**只有朋友确认资源配置和预算、4090 预检通过，正式队列才可能启动。** 当前朋友服务器预算、CPU、RAM、磁盘尚未实测；其 GPU 真正运行状态标为 `PENDING`。预检本身会占用少量授权 GPU 时间。

## 22 项固定任务

| 组 | ID | 次数 | 来源 |
|---|---|---:|---|
| A | `D17_F01`–`D17_F05` | 5 | B01 实际 DenseNet-121 配置 |
| A | `C17_F01`–`C17_F05` | 5 | M02 实际 ConvNeXt-Tiny 配置 |
| A | `S17_F01`–`S17_F05` | 5 | T01 实际 Swin-Tiny 配置 |
| N | `N17_F01`–`N17_F05` | 5 | N01 EfficientNetV2-S，训练中同时维护 EMA |
| P | `CS17_F01`、`NM17_F01` | 2 | ConvNeXt-Small、EfficientNetV2-M 的 Fold 1 Pilot |

三骨干原 15 项配置和冻结患者五折不变。每折只用另外四折训练；官方 Val 仅用于早停和最佳 checkpoint，当前折仅作 OOF；Test 完全无标签，只推理。Pilot 单折 OOF 不冒充完整五折 OOF。额外四折补齐、其他随机种子和蒸馏仍禁用，不会由 22 项结果自动触发。EMA 是训练内附加分支，单独保存其最佳权重和 Val/OOF/Test 概率；早停仍监控原始权重 Study Macro-AUC。

## 自动化与状态

队列动态分配空闲卡，子进程各自设置 `CUDA_VISIBLE_DEVICES`，每卡最多一个任务。配置 `CXR_MAX_CONCURRENT` 控制并发；启动前核对 GPU 型号/显存、CPU 核数、可用内存、输出空间和 ZIP 随机读取速度，资源不足则明确报错，**不静默改训练 batch 或模型参数**。预检还检查 Python3.12、PyTorch2.10.0、torchvision0.25.0、CUDA12.8、固定分割、AMP、真实前反向与断点恢复。失败任务保留 `FAILED.json`、错误日志和最近可恢复 checkpoint；有限重试只用于判定为可恢复的 I/O 类错误。OOM、非有限梯度、哈希错误不会自动改配置重跑。

每个成功模型立即在 `outputs/phase17_batch/experiments/<ID>/` 保存纯 `state_dict` 的 `best_model.pth`、SHA、实际配置、逐 Epoch 历史、日志、Val/OOF/Test Study 概率、逐类指标、环境、模型大小、推理研究测速、部署状态、恢复 checkpoint 和 `DONE.json`；任务状态同步写入 `reports/experiment_status.csv`。Test 缺失或其推理失败时，已经完成的训练/Val/OOF 保留，状态为 `test_pending`/`test_failed`；补齐 Test 后用相同启动命令 `--resume` 路径完成 Test，**不会重训已完成模型**。

五折齐全时按 ID 而非 CSV 行号合并完整 OOF、官方 Val 和无标签 Test 概率。固定 E17 公式为 `0.25×D5+0.25×C5+0.5×S5`，不搜索权重。Test 合格时自动写四列、每 Study 全 10 类的候选 CSV 和 manifest，**不会自动上传比赛平台**。若一个 Fold 缺失，对应五折 Test/OOF 标记 `INCOMPLETE`，不生成假完整结果。

查看和安全停止：

```bash
source .venv/bin/activate
python scripts/batch_run.py --config configs/batch_plan.yaml --action status --root "$CXR_OUTPUT_ROOT/phase17_batch"
python scripts/batch_run.py --config configs/batch_plan.yaml --action stop --root "$CXR_OUTPUT_ROOT/phase17_batch"
```

`stop` 只阻止新任务，正在运行的任务按各自超时/完成逻辑结束。再次启动前须确认原队列不在运行并处理其 STOP/lock，不能强行解锁活跃队列。服务器若使用 Slurm 等调度器，请先让管理员确认实际分配的 GPU ID 和 CPU/RAM，修改本地授权配置；不要同时运行两个队列共享一个输出目录。

## 结果直接交付

结果一直保存在 `outputs/phase17_batch/{manifests,experiments,oof,validation,test_predictions,analysis,deployment,submissions,reports}`。没有 ZIP/TAR/GZIP 打包；不自动删除服务器文件。`manifests/delivery_manifest.json`、各模型 `batch_frozen_manifest.json` 记录 SHA-256。完成后把整个 `phase17_batch/` 文件夹通过双方同意的大文件传输渠道交给用户，或提供授权下载位置。仓库不接受胸片数据、checkpoint 或提交 CSV。Windows 导入与核验见 [README_results.md](README_results.md)。

## 部署硬约束

最终全部推理权重压缩前总大小 ≤500,000,000 字节；按照用户更严格的边界，完整单图端到端延迟 **<100ms**。4090 的研究测速不能证明目标评测硬件（可能是 T4）合规；官方硬件、精度、batch、起止范围和统计量尚缺完整原文。因此未经官方同口径实测均为 `LATENCY_UNVERIFIED`，超过大小为 `SIZE_FAIL`，已证实延迟不合规为 `LATENCY_FAIL`，只有准确率、大小和官方时延都核实才为 `PASS`。单模型的 Forward 时间不能代替完整融合测速。FP16、量化和蒸馏后的实际产物都需重新验证 Val AUC、大小和时延。

待官方给出完整目标计时口径并在目标设备运行时，可使用 `python scripts/benchmark_target_latency.py --config configs/batch_plan.yaml --root "$CXR_OUTPUT_ROOT/phase17_batch" --candidate D17_F01 --official-protocol /authorized/official_timing_protocol.json`。脚本仅在设备、精度、批量、计时范围和统计量逐项吻合时输出官方证据；否则仅存研究测速，不标记 `PASS`。该协议文件必须由用户审核并按脚本的规范化 JSON 规则固定 SHA-256，目前尚未具备。

官方 torchvision 权重文件：DenseNet `densenet121-a639ec97.pth`；ConvNeXt-Tiny `convnext_tiny-983f1562.pth`；Swin-Tiny `swin_t-704ceda3.pth`；EfficientNetV2-S `efficientnet_v2_s-dd5fe13b.pth`；Pilot 分别为 `convnext_small-0c510722.pth`（SHA-256 `0c510722adfd92966a2bd72b92f785ca05966bbac03cafe2f7a90b1f54bfab9a`）和 `efficientnet_v2_m-dc08266a.pth`（SHA-256 `dc08266a3779e28992d6a42528a32453f6361d0e4e247b5f20f45e13e7add499`）。六者完整 SHA 均在计划中固定；朋友需通过官方来源或单独授权传输取得权重，并由预检逐字节核对。权重文件不提交 Git。
