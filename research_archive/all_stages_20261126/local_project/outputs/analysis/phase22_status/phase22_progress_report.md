# Phase 22 当前进展（2026-10-21）

## Fold3

- 本地 RTX 4060 Laptop 的 D17-F3 已启动并在训练。Fold3 启动器按 D17、C17、S17 顺序执行，每个模型完成后独立重算官方 Val、Fold3 OOF 并冻结 SHA-256。Fold4–5 不在启动器中。
- 启动前复核 Fold1–2 六个冻结目录：`best_model.pt`、`frozen_manifest.json`、官方 Val 与 OOF Study 概率均存在。三个 Fold3 配置与 Fold2 对应配置除实验名和折选择外一致。患者、Study、图像五折索引 SHA-256 与配置一致；ZIP 存在。
- 第一次启动在 Windows 沙箱内的 DataLoader 子进程创建时被 `WinError 5` 拒绝，未完成训练 batch，日志独立归档到 `outputs/experiments/d17_f3_20261021_startup_failure_sandbox/`。随后以同一冻结配置在允许子进程的环境中启动成功；未改动超参数。
- 固定融合和前三折部分 OOF 的 CPU 计算脚本已建立。Fold3 三个模型冻结后，后台观察脚本会运行固定权重 `0.25 D17 + 0.25 C17 + 0.50 S17`；前三折 OOF 将按患者 ID 联合重新计算，绝不把单折 AUC 算术平均作为完整 OOF。
- 当前结果以 `status.json` 和训练目录的 `metrics.jsonl` 为准。未完成的 Fold3 Val/OOF AUC 均为待定。

## 朋友 26 项

状态为 **UNKNOWN**。本机没有合法同步的朋友服务器正式结果目录；未访问朋友服务器，也未改变私有 GitHub 分支或远程队列。已准备 `scripts/import_phase17_friend_results.py`，收到直接文件夹后核对 Git commit、官方数据/划分哈希、训练图像索引、每模型 checkpoint SHA-256、859 Val / 1414 Test / 各折 OOF 覆盖以及患者隔离。通过后才复制到 `outputs/phase17_friend_import/<version>/`；存在相同版本时拒绝覆盖。`scripts/analyze_phase17_friend_import.py` 可在 CPU 上重算单模型、完整五折 OOF、预定 E17 Val 融合和患者级 bootstrap，输出逐类指标与部署大小表。真实远程目录尚未交付，这些全量导入路径仍待真实产物验收。

## 部署与候选

- `outputs/analysis/phase22_deployment/` 独立复用 Phase20 已完成全 Val 一致性验证的 E03/E08 共享预处理实现。CPU verify 模式通过 E03/E08 冻结 checkpoint 哈希和权重大小检查。T4 benchmark 模式只提供可复现测量程序；未在 T4 运行，也缺官方完整计时口径，因此状态保持 `LATENCY_UNVERIFIED`。
- 现有候选表：`outputs/analysis/phase22_deployment/submission_candidates.csv`。E08-EMA 是冻结官方 77.75 分基准；E17-F1 是单折候选；Fold3/完整五折和远程 N/G/H 均未生成正式新候选。没有计算隐藏 Test AUC 或提交排行榜。

## 本轮轻量校验

- 新增 Python 脚本静态编译通过。
- 导入器概率列顺序、ID 覆盖、重复、非有限值和越界值的 6 个 CPU 合成测试通过。
- Fold1、Fold2 固定融合 OOF 在同一读取脚本中复算为 0.8068045、0.8079803，符合已有记录。
- 没有运行额外 GPU 测试；正在训练的 D17-F3 保持独占 GPU。
