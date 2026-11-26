# Phase 19：弱类、延迟与可部署候选审计

日期：2026-10-21。**本轮未训练、未改冻结权重/预测/提交、未连接朋友服务器、未读取隐藏 Test 标签。**可复算代码：`analyze_phase19.py`、`benchmark_e08_segments.py`；输入为冻结 E07/E08 Val 概率、官方原始 ZIP 的 `valid.csv` 与少量合法 Val 图片、既有 manifest/checkpoint。官方 Val 共 859 Study、228 患者。

## 结论 1：弱类

| 类别 | 阳性/阴性 Study | 阳性/阴性患者¹ | E08 AUC | E08 AP | E08−E07 AUC |
|---|---:|---:|---:|---:|---:|
| 8 Lung Opacity | 214 / 645 | 72 / 156 | 0.72397 | 0.43321 | -0.00064 |
| 1 Enlarged Cardiomediastinum | 24 / 835 | 16 / 212 | 0.72824 | 0.10459 | -0.01233 |
| 4 Pneumonia | 68 / 791 | 37 / 191 | 0.75072 | 0.23297 | +0.00230 |

¹ 阴性患者指该类没有阳性 Study 的患者；同一患者可有多个 Study。类别 8 阳性数最多且 AUC 最低，是首要系统性误差目标。类别 1 仅 16 名阳性患者，患者级 1000 次有效 bootstrap 的 E08 AUC 95% CI 为 [0.5524, 0.8619]，不宜把小幅差异当稳定规律。类别 4 的阳性中 36/68 与类别 8 共现、10/68 与 Consolidation 共现，提示需要审阅报告标签语义，但未证明标签错误。来源：`outputs/analysis/phase18_leaderboard_diagnosis/e07_e08_per_class.csv`、本目录 `weak_label_cooccurrence.json`、`bootstrap_comparisons.json`。

`study_view_stratification.csv` 显示类别 8 单图/多图 AUC 0.6593/0.7157，类别 4 为 0.6691/0.8200，类别 1 为 0.6016/0.8388；组内阳性数、患者数和病例构成不同，不可据此断言聚合导致性能差。`weak_class_error_analysis.csv` 保存每个弱类 Study 的预测差和正负排序错误比例，`local_image_quality_review.csv` 记录 18 张只读抽样图片可读性、宽高、黑边与像素分布。未对医疗图像作临床诊断，未上传图像。

## 结论 2：E08 完整流程延迟

本地 RTX 4060 Laptop、固定一张官方 Val 图像、384×384、批量 1、FP16 autocast、预热 5 次、重复 16 次、每段 CUDA 同步，模型已加载在 GPU；从 ZIP 读取到五模型概率融合及单图 Study Mean 的**完整均值 175.57ms，P50 175.02ms，P95 197.51ms，最大 201.68ms**。解码均值 31.93ms，等比例缩放填充 36.91ms，五个模型各自前向约 22.04/22.30/21.37/9.72/23.86ms，融合及 CPU 回传约 0.30ms。分段同步开销不同，分段时间不可机械求和为完整时间。来源：`latency_breakdown.csv`、`latency_benchmark.json`、`benchmark_e08_segments.py`。

先前 156.7ms 是另一组 Val 图片与计时实现的均值；两者均为本机研究测量。官方原文仅给出单图端到端 ≤100ms 与模型总大小 ≤500MB；目标硬件、精度、预热、批量、计时起止点、统计量等完整口径未取得可信原文。公开检索未找到与本赛事对应的详细规范。**因此所有候选仍为 `LATENCY_UNVERIFIED`，4060 或朋友 4090 的速度都不能证明 T4/官方环境合规。** E08 权重 278,436,014 字节，满足大小上限，但不能凭此标记整体合规。五模型离线 Test CSV 的缓存不能代替真实新图像完整推理。参见 `preprocessing_audit.md`。

## 结论 3：有限轻量候选

| 冻结候选 | 模型数 | 权重 MB（十进制） | Val Macro-AUC | 相对 E08 AUC 与患者配对 95% CI | 目标设备延迟 |
|---|---:|---:|---:|---|---|
| E08-EMA | 5 | 278.44 | 0.81167 | 基准 | 未测 |
| E07 | 5 | 307.18 | 0.81152 | -0.00015 [-0.00704,+0.00750] | 未测 |
| E03 | 4 | 196.77 | 0.80993 | -0.00174 [-0.00753,+0.00380] | 未测 |
| E01 | 3 | 85.39 | 0.80267 | -0.00900 [-0.01682,-0.00023] | 未测 |
| N01_EMA | 1 | 81.67 | 0.80491 | -0.00676 [-0.01076,-0.00208] | 未测 |
| M02 | 1 | 111.38 | 0.79724 | -0.01442 [-0.02515,-0.00641] | 未测 |

所有数值来自同一 859 Study 的冻结概率，以 `Subject_id,Study_id` 严格对齐，固定原融合公式，无权重搜索。配对 bootstrap 以 228 患者为单位，种子 20261021、1000 次有效抽样、0 次无效；若任何类别缺正/负 Study 则丢弃并重抽。CI 未对多重比较修正，也不能消除官方 Val 的反复选择偏差。来源：`lightweight_ensemble_candidates.csv`、`bootstrap_comparisons.json`、`outputs/analysis/phase18_leaderboard_diagnosis/val_selection_bias_audit.md`。

**优先进行目标硬件测试的已有轻量融合是 E03。** 它较 E08 少一个 EfficientNetV2-S 分支，Val 差异区间跨 0；但它仍有四个模型，不能预设能达到 100ms。单模型 N01_EMA 可作为速度对照，Val AUC 则有明确下降信号。建议先优化共享解码/预处理和目标口径测速，当前不启动新训练；等待远程完整五折 OOF 与蒸馏结果再决定模型组合。详见 `next_experiment_decision.md`。

## 已证实与待证实

- 已证实：现有预处理保比例填充、18/18 抽样图片可读且尺寸匹配；E08 的类别 8/1/4 弱；本地五模型完整流程超过 100ms；大小低于 500MB；Fold1/2 部分 OOF 与冻结划分匹配、患者不交叉。来源：`preprocessing_audit.md`、`latency_breakdown.csv`、Phase18 `oof_integrity_report.json`。
- 待证实：官方目标环境延迟、图像或标签质量的因果问题、远程 26 项运行状态、完整五折 OOF 稳定性。远程状态只能列 `UNKNOWN`，来源：`remote_batch_status.md`。
- 原样保留：E08 官方用户报告 77.75、E07 76.65、榜首 80.65。这些分数不能推断隐藏 Test 的逐类 AUC，也不能由一次排名变化证实 EMA 的因果收益。
