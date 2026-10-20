# Phase 17-Batch 26项无人值守计划验收（2026-10-20）

## 结论

默认计划为 22 个基础模型加 G/H 四个全 Train 模型，单次 `friend_start.sh` 启动会按依赖顺序执行。**本地 CPU/合成数据验收通过；4090 真实运行仍为 PENDING。** 朋友必须明确授权预检、26项总 GPU 小时及 G/H 子预算，并先取得强制 `REMOTE_PREFLIGHT` 通过凭证。真实 G/H 单步 AMP 预检在基础折结果齐全后逐项执行；失败的任务保持 BLOCKED。代码不会因本报告而在本机启动训练。

## 任务与依赖

| 组 | 任务 | 数量 | 运行前条件 |
|---|---|---:|---|
| D17/C17/S17 | 三骨干各五折 | 15 | 冻结五折、合法ImageNet权重、远程预检与预算 |
| N17 | EfficientNetV2-S五折，附带EMA | 5 | 同上 |
| Pilot | ConvNeXt-Small、EfficientNetV2-M Fold1 | 2 | 同上；失败不阻断G/H |
| G17 | OOF规则所选两个骨干全Train重训 | 2 | D/C/S/N共20折完成、完整OOF、预定筛选、GPU短步预检 |
| H17 | DenseNet与EfficientNetV2-S蒸馏 | 2 | 完整患者隔离教师OOF、固定蒸馏参数、GPU短步预检 |

G/H Epoch上限取对应五折最佳Epoch中位数；教师固定为 `0.25×D17_OOF + 0.25×C17_OOF + 0.50×S17_OOF`。多图Study的教师软标签按每Study图像数倒数加权。G/H没有独立OOF，指标与概率文件必须标N/A，不生成伪OOF。Train/Val/Test用途维持原协议。

## 本地测试证据

- 全套仅CPU回归：**65 passed、0 failed、3 skipped**。日志在本地忽略目录 `outputs/phase17_batch_26_acceptance_20261020/verified.log`，JUnit记录在同目录 `verified.xml`。
- 合成依赖及G/H专项：**27 passed**；覆盖一次启动22+4、预算分配、Pilot失败不阻断G/H、单任务GPU短测失败和恢复、患者隔离、固定Epoch、软标签维度/权重、Val/Test概率、14140行Submission、候选去重和隐藏Test AUC为N/A。
- 3项跳过均需CUDA：1项档案推理、2项AMP断点恢复参数。`REMOTE_PREFLIGHT` 在朋友4090上要求三项全部实际通过，跳过亦失败。
- 尚未实测朋友的8×4090吞吐、显存、磁盘、真实G/H全Epoch训练、官方目标设备单图端到端时延。不得把CPU验收当作这些项目的PASS。

## 产物与安全门禁

每个成功任务立即写入独立 `experiments/<ID>/` 的最佳推理权重、恢复状态、配置、日志、Val/Test概率、适用OOF、逐类指标、SHA、研究性4090延迟与DONE；失败或阻塞保留既有产物。完整五折OOF/Val/Test及固定融合结果写入分类目录，不压缩、不自动上传。官方Val曾反复用于选择；隐藏Test只有概率与格式合格的Submission候选，没有Test AUC。

最终部署要求全部推理权重≤500,000,000字节，按官方目标环境完整单图计时≤100ms。80ms仅为内部安全目标。只有4090研究测速或缺少官方计时细则时状态为`LATENCY_UNVERIFIED`，不声称合规。

GitHub发布仍需用户明确批准；代码仓库不包含原始胸片、私有预训练权重、checkpoint或访问凭证。本地Phase17冻结实验未修改。
