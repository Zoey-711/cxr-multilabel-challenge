# H17-02 batch=1 官方 77.7 候选审计（2026-10-29）

## 结论与状态

将 `outputs/submissions/H17_02_BATCH1_20261024/submission.csv` 作为当前**重点轻量模型基线**。官方 77.7 分由用户确认；本机没有主办方成绩回执，因此标记 `USER_CONFIRMED`，不把分数当作本地 Test AUC。旧 H17-02 官方 77.50 分亦来自用户先前报告。历史 Submission、manifest、checkpoint 均未修改。

新候选的模型是单个 `H17_DISTILL_02` EfficientNetV2-S，384 输入，`black_pad_v1`，ImageNet 归一化，batch=1，FP16 autocast，图像 sigmoid 后按 Study Mean 聚合。提交目录 manifest 指向 Phase31 独立部署包。已逐一重算该包 manifest 所列 **63 个文件**的大小和 SHA-256：缺失 0，变更 0。推理权重 `model/H17_02.pth` SHA-256 为 `35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69`，与朋友结果中 raw `best_model.pth` 的 manifest 一致；来源训练结果 commit `cbfc5997438f371086cd1fbe232890590cd67ffb`，训练代码 commit `8b7944d744f91e753b912367adeb354e8b7e21d5`。训练为全 Train 蒸馏，不能声称有学生独立 OOF。

本地 batch=1 FP16 的 859 Study Val Macro-AUC 是 **0.8086102168**。旧朋友提交 manifest 的 Val AUC 是 **0.8086139109**。新 CSV SHA-256 为 `00e093b43fecc115708568cc0e47c667863e6963ab1fa4d0147f8fbed5930aac`；旧 CSV 是 `1bd7a6deae6d716b9ae4e08dd29d2cb1a83ae16b9d49c6fa016629810d88f30e`。新文件通过 1414 Study × 10 类、14,140 行、概率范围及重复键检查；随包 `predict.py` / `cost.py` 共用 `predict_image()`，已有同入口 100 个 Test 单图 Study 自检最大差 0，完整宽表与 Submission 最大差 0。Test 标签未使用，Test AUC 不适用。

## 与旧 77.50 版本比较

旧 `results/phase17_batch/submissions/single_models/H17_DISTILL_02_submission.csv` 与其 `experiments/H17_DISTILL_02/test_predictions.csv` 逐项完全相同。新旧 CSV 的 14,140 个 `(Subject_id, Study_id, Predict_class)` 键完全相同；**7053 行概率改变**，平均绝对差 `0.0000677373`，P99 `0.00048828125`，最大差 `0.0009765625`（`subject_12097/study_22351` 类别 9）。两版 Val AUC 差约 `-0.0000036941`。源数据见 `submission_version_comparison_v3.json`。

旧 Test 概率 provenance 绑定封装的 `best_model.pt`（SHA `9b5fc2...`），旧提交 manifest 与新部署包绑定从封装 checkpoint 导出的纯 `state_dict` `best_model.pth`（SHA `35fb17...`）。原训练导出代码含逐张量比较，但旧封装 `.pt` 未同步至本机，故本轮不能独立重验两种文件的张量完全相同；文件哈希不同本身也不表示模型参数不同。旧概率在朋友 RTX4090 产生，新概率在本机 RTX4060 用 batch=1 固定入口产生；硬件数值环境、批量与推理代码路径均可能影响 FP16 舍入。**不能证明官方 77.50→77.7 仅由 batch=1 造成**，也不能凭隐藏 Test 榜单差异判定因果。若要隔离 batch 因素，须在同一设备、同一输入和同一代码/精度下只改变 batch，并比较概率；这不应使用 Test 标签。

## 本轮重新测得的端到端性能

本机 RTX4060 Laptop，Python 3.12.4、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128，使用已冻结的 `predict.py` 与 `cost.py`。固定样本是 Phase29 预先选定的**官方 Val 影像**，5 张预热、100 张正式图、batch=1、连续 3 轮。每张计时包含读取文件、解码、预处理、CPU→GPU、前向、概率后处理和 CSV 写出；模型加载和预热不计入，计时前后均同步 CUDA。每张先取三轮均值，再对 100 张统计。

| 4060 指标 | 本轮实测 |
| --- | ---: |
| Mean | **93.2946 ms** |
| P50 | 94.0668 ms |
| P95 | 102.3416 ms |
| Max | 114.7912 ms |

完整机器可读证据为 `cost_4060_20261029/cost_report.json` 与 `cost_predictions.csv`。这是重新执行的测量，不是沿用 Phase30 的 88.46 ms。Val 样本不是主办方固定 Test 计时图；本轮计时报告因此正确标记 `official_sample_attested=false` 和 `T4_UNVERIFIED`。P95 超过 100ms 并不直接触发 PDF 中“100 图平均”判定，但提示运行余量有限。

## 大小与部署判断

实际加载的权重 81,651,999 字节；连同必要部署配置 81,654,915 字节，完整部署包（不含结果）81,876,207 字节，均明显低于 **500,000,000 字节**，大小 `PASS`。官方规则 PDF 第 7 条要求指定硬件、固定 5+100 图、batch=1、三轮读图到结果写出；第 8 条要求实际权重及必要配套文件解压后不超过 500MB。PDF 本地副本：`outputs/submissions/E08_ONNX_FP32_20261028/official_rules_20261023.pdf`。本机无 T4，也无主办方固定的 5+100 计时图，**T4 ≤100ms 未验证**。4060 均值仅比 100ms 低约 6.7ms；T4 上存在超时风险，不能据此标记部署合规。

本候选可作为后续轻量化冲榜的**冻结参考基线**：固定此 CSV SHA、权重 SHA、推理精度、batch、预处理和 Study 聚合。新的优化必须独立产物、Val/大小/延迟重验。复赛正式材料仍需 T4 实测、官方固定图与 Study 映射的一致性核验，以及预训练权重再分发许可和朋友数据使用授权的人工确认。
