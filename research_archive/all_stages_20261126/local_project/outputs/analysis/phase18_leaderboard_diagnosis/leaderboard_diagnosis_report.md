# Phase 18：排行榜诊断（只读统计）

## 范围和复现

本报告仅使用已冻结的 Study 级验证概率、患者五折索引、实验 metadata、checkpoint 文件大小及用户提供的官方成绩。未读取隐藏 Test 标签、未推理 Test、未训练模型、未改动任何原实验文件。重算命令（CPU）：`python outputs/analysis/phase18_leaderboard_diagnosis/reproduce_analysis.py`。脚本源文件、`computed_summary.json` 和各 CSV 提供计算方法及数据路径；相应输入 SHA 可在冻结 manifest 中追溯。

官方成绩为用户报告值：E01 75.51、E03 76.45、E07 76.65、E08-EMA 77.75；它们不是从本地隐藏标签计算的。若官方显示值为百分制 Macro-AUC，E08 对 E07 的榜单差为 0.0110，而本地 Val 差仅 0.000149。榜首用户报告为 80.65，目前还差 2.90 个榜单分点。一次榜单变化不能说明原因。

## 统一实验清单

`experiment_inventory.csv` 扫描到 19 个已有完整骨干推理 checkpoint/EMA 分支、4 个冻结特征聚合头 V00–V03，以及 6 个已定义融合候选，共 29 行。表中逐项记录架构、配置、最佳 Epoch、官方 Val Macro-AUC/十类 AUC/AP/阳性数、实际 OOF 覆盖、是否复用官方 Val、训练范围、预处理、文件字节数、SHA 与官方成绩来源。V00–V03 的部署体积同时包含 M02 主干和聚合头；`last_training.pt` 等优化器恢复状态未被当成推理权重。不存在的 OOF 明确标为 N/A。若同一目录内还有失败/预检临时产物，它们不被伪装成完整训练模型。

阶段性结果：E07 Val 0.8115193，E08-EMA Val 0.8116686；E17-F1 Val 0.8068890、Fold1 OOF 0.8068045；E17-F2 Val 0.79890、Fold2 OOF 0.80798。Fold1+2 固定融合的合并部分 OOF 为 0.8057034，覆盖 7,418/18,513 Train Study；它不是完整五折 OOF。来源：`oof_integrity_report.json`、各实验 `validation_analysis` 和 `oof_analysis` CSV、Fold2 `fold2_summary.json`。

## E07 与 E08-EMA：同一 859 Study 的逐类比较

E07 按冻结公式 `E07=0.5 E03+0.5 T01` 重建；E08 按 `E08=0.5 E03+0.5 N01_EMA` 重建，且与已保存 E08 Val CSV 逐值一致。两者都与 B01 验证标签的 Subject_id、Study_id、类别 1–10 完全对齐。每类 AUROC/AP 由 `sklearn.metrics` 在 Study 级计算；排序错误互补以所有阳性×阴性 Study 对计数，严格比较概率次序并单独记录平局。详细数值见 `e07_e08_per_class.csv`、`figures/`。

| 类别 | 阳性 Study（患者） | E07 AUC | E08 AUC | E08−E07 | E08 AP |
|---|---:|---:|---:|---:|---:|
| 1 Enlarged Cardiomediastinum | 24（16） | 0.74057 | 0.72824 | -0.01232 | 0.10459 |
| 2 Pneumothorax | 37（13） | 0.86089 | 0.88387 | +0.02298 | 0.36588 |
| 3 Consolidation | 46（24） | 0.78298 | 0.78274 | -0.00024 | 0.16994 |
| 4 Pneumonia | 68（37） | 0.74842 | 0.75072 | +0.00230 | 0.23298 |
| 5 Edema | 115（34） | 0.87973 | 0.87861 | -0.00112 | 0.44911 |
| 6 Cardiomegaly | 177（66） | 0.78411 | 0.77554 | -0.00856 | 0.45427 |
| 7 Atelectasis | 193（64） | 0.80206 | 0.79815 | -0.00391 | 0.53151 |
| 8 Lung Opacity | 214（72） | 0.72462 | 0.72397 | -0.00065 | 0.43321 |
| 9 Pleural Effusion | 260（52） | 0.90907 | 0.91440 | +0.00532 | 0.80112 |
| 10 No Finding | 306（166） | 0.88274 | 0.88044 | -0.00231 | 0.83743 |

E08 的三个最低 AUC 类别是 **8、1、4**。点估计上类别 2、4、9 改善，其中类别 2 最明显；类别 1 和 6 退化。类别 2 差值的未做多重比较校正的患者 bootstrap 95% 区间约 [+0.00693,+0.04727]，类别 4/9 区间均跨 0。十类 Macro 差 +0.000149 的患者级配对区间 **[-0.007385,+0.007402]**，故不能把 E08 的 Val 总体差异称为统计可靠。来源：`e07_e08_bootstrap.json`（2000 有效抽样、固定种子 20261021、0 无效）。

![E07 与 E08 逐类 AUC](figures/e07_e08_auc.png)
![E08 相对 E07 的 AUC 差](figures/e08_auc_delta.png)
![阳性数与 E08 AUC](figures/support_vs_auc.png)

## 现有骨干互补性

所有相关性与逐类排序错误均在同一 859 Study 上计算，完整矩阵见 `model_complementarity.csv`。有限、预先存在的对比见 `fixed_ablation.csv`：E01→E03（加 M02）Val +0.00726；E03→E07（加 T01）+0.00159；E03→E08（加 N01_EMA）+0.00174。N01 原权重→EMA 单模型 +0.01232，但这是两份不同最佳 Epoch 权重的对照，不等于 EMA 操作的单独因果效应。E07 与 E08 逐类预测 Pearson 均值 0.948，说明大部分预测高度相近；类别 1 的相关性相对较低（0.860）。

在现有 Val 上最清楚的增量互补是 **DenseNet 组合 E01 与 ConvNeXt M02**：两者逐类 Pearson 均值 0.804，固定融合 +0.00726；M02 与 Swin T01 均值 0.800，可产生不同排序，但 E03→E07 的实际 AUC 增量较小。E03 与 N01_EMA 均值 0.879，固定融合 +0.00174。相关性只是预测差异，必须结合 AUC 差和患者 bootstrap 判断收益；本报告不进行网格权重搜索。当前不存在 E08 或 E07 对 Train 患者的独立 OOF，不能用它们与 Phase17 OOF 做同一患者的配对推断。

## 泛化误差与数据协议

证据、影响边界和建议逐项见 `val_selection_bias_audit.md`。关键风险：

- 官方 Val 被多次用于 Early Stopping 和方案选择；选出的最高 Val 不具有全新测试证据的含义。
- 数据审计未发现标识符级 Train/Val/Test 患者或 Study 泄漏、重复图像关联、No Finding 共现或显式不确定标签编码；但未做全量像素/感知哈希去重。官方“完整多标签”规则把未列出的类别视为竞赛阴性目标，不等于临床确认阴性。
- Fold1/2 六份 OOF 与冻结患者/Study/图像折索引一致，且两折相互无交叉；Fold3–5 缺失。不同折的单折 AUC 不可平均替代整体 OOF。
- 验证和测试采用相同确定性 RGB→保比例缩放/黑边填充→ImageNet 归一化→逐图 sigmoid→Study Mean；训练随机增强仅作用于训练集。现有审计无解剖区域裁切；TTA 未使用。来源：`src/cxr/data/dataset.py`、`outputs/analysis/phase6_preprocessing/`、实验配置及提交 manifest。
- 无标签 Test 的单图比例 54.8%，Val 45.5%；AP 图像比例 Test 49.1%，Val 41.1%。这是可观测元数据差异，不能推断隐藏 Test 逐类标签或哪一类导致官方分数变化。来源：`outputs/analysis/test_shift_20261016/test_shift_audit.json`。

## 部署硬约束

`deployment_feasibility.csv` 统计实际推理文件字节数；E08-EMA 五个 checkpoint 合计 **278.436 MB**，E07 合计 **307.175 MB**，均满足 ≤500 MB。**延迟并未证明合规**：E08 五模型在本机 RTX 4060 Laptop 的已有完整单图流程测试均值约 **156.7ms**（30 张，含 ZIP 读取、解码、预处理、传输、前向、sigmoid、回传）；这已超过 100ms，本机结果也不能替代官方 T4 或等效环境测试。官方要求是单张图像端到端 **≤100ms**，最终候选必须完整推理图、目标硬件、官方批量/精度/统计口径实测；否则保持 `LATENCY_UNVERIFIED`。来源：N01 `inference_benchmark.json`、T01 `inference_benchmark.json` 和本报告部署表。完整 15 模型教师集成只适合研究，不能凭 AUC 忽略尺寸与时延。

## 下一步结论

**不建议现在新增训练。**按 `prioritized_experiment_plan.md` 先完成零训练成本的弱类错误审阅与目标硬件部署测速，再等待已批准的远程 26 项产生完整患者级五折 OOF。其后只评估预先冻结的有限融合、少模型部署与 H17 学生；若证据仍不足，再决定唯一受控追加实验。E08、E07、已有提交和全部冻结文件保持原样。
