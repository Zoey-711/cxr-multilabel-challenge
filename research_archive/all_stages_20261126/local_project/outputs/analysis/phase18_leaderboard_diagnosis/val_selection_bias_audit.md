# 官方 Val 复用与泛化风险审计

## 已核实的事实

1. 官方 Val 为 859 个 Study、228 名患者。B01、B02、B03、M02、T01、N01 以及 Phase17 各 Fold 均使用同一官方 Val 的 Study Macro-AUC 进行最佳 Epoch 选择；研究过程中还比较过损失、增强、骨干、分辨率、采样和融合。**因此 Val 的候选排名不是一次全新的独立测试。**来源：各实验 `config.yaml`、`best_metrics.json`、`validation_analysis/study_predictions.csv`；统一清单见 `experiment_inventory.csv`。
2. E07 与 E08-EMA 的 859 个 Study/Subject ID 和十类标签完全对齐。E07 按冻结公式 `0.5 E03 + 0.5 T01` 从原始逐 Study 概率重建为 0.8115193038；E08-EMA 按 `0.5 E03 + 0.5 N01_EMA` 重建为 0.8116686461，并与已冻结 `outputs/submissions/E08_EMA_20261020/val_predictions.csv` 的最大绝对差小于 1e-12。来源：`reproduce_analysis.py` 的对齐断言、`computed_summary.json`、E07 `fusion_config.json`、E08 `manifest.json`。
3. 以 Subject_id 为抽样单位、同一患者的全部 Study 一起重采样，固定种子 20261021，得到 2000 次有效配对 bootstrap；若任一类缺少正或负 Study，则整次抽样丢弃并补抽。本次无无效抽样。E08−E07 Macro-AUC = +0.000149，95% 百分位区间约 [-0.007385, +0.007402]，**不能称为可靠改善**。来源：`e07_e08_bootstrap.json`。
4. 官方数据审计记录 Train/Val/Test 的 Subject 和 Study ID 两两重叠均为 0；重复图像关联和重复图像 ID 为 0；No Finding 与疾病类别共现计数为 0；未观察到显式不确定编码和标签空白行。这里的“未列出类=竞赛阴性目标”取决于官方完整多标签标注规则，不能推断临床报告中绝对不存在疾病。来源：`outputs/audits/data_audit.json`。该审计检查标识符与关联记录，**未证明不存在像素级或近重复图像**。
5. 冻结的患者五折索引含 11,055 名患者、18,513 个 Study、30,344 张图。Fold1/2 的六份 OOF 文件各自与相应折的患者、Study、图像索引一致；Fold1 与 Fold2 之间 0 患者、0 Study 重叠；其合并 OOF 只覆盖 7,418 Study，不能称为完整五折 OOF。来源：`outputs/analysis/phase17_patient_folds/{patient,study,image}_folds.csv`、各 `*_f1_*` 和 `*_f2_*` 的 `oof_analysis/study_predictions.csv`、`oof_integrity_report.json`。
6. 训练和确定性验证/测试使用 384×384 的 RGB 转换、保比例 `ImageOps.pad` 黑色填充、ImageNet 归一化。随机旋转、平移和亮度/对比度增强仅用于适用模型的训练；验证/测试无随机增强，冻结候选未使用 TTA。早期预处理审计未发现裁切源像素。来源：`src/cxr/data/dataset.py`、`outputs/analysis/phase6_preprocessing/{report.md,audit.json}`、M02/T01/N01 配置与 E07/E08 manifest。单图视角和数据源差异仍可能导致模型误差。
7. 无标签 Test 的可观测元数据与 Val 不完全一致：Test 单图 Study 775/1414（54.8%），Val 391/859（45.5%）；Test 图像 AP 1090/2218（49.1%），Val 584/1422（41.1%）；图像高度中位数 Test 2778、Val 2960。**这是元数据分布差异，不是已证实的隐藏标签漂移。**来源：`outputs/analysis/test_shift_20261016/test_shift_audit.json`。
8. N01 的普通和 EMA 权重在本机使用同一推理代码，但不同设备与精度的数值误差尚未跨环境逐 Study 比对。官方 T4 端到端时延也未测量。来源：`outputs/experiments/n01_efficientnet_v2_s_384_amp_retry_20261018/inference_benchmark.json`、`deployment_feasibility.csv`。

## 对排行榜差距的限定解释

用户报告 E07 为 76.65、E08-EMA 为 77.75。若平台分数是百分制 Macro-AUC，则隐藏 Test 的点估计改善为 0.0110，而 Val 仅为 0.000149。**可行但未证实**的解释包括：859 个 Val Study 的有限样本波动、重复使用 Val 造成的选择偏差、Test/Val 可观测视角与单图比例差异，以及两个骨干在不同样本上的排序互补。没有隐藏 Test 标签，无法判定哪一项造成差距，也不能计算隐藏 Test 逐类 AUC。一次榜单变化不能作为因果证据。

## 处理原则

- 冻结 E08 作为当前正式基准；不依据排行榜反复拟合融合权重。
- 不再用 Phase15 内部 Holdout 选择新候选；它已被评估，见 `outputs/analysis/phase15_formal_20261018/`。
- 后续以完整患者级五折 OOF 检查稳定性；官方 Val 用于既定模型选择和辅助对照，不当作新的独立泛化证明。
- 部署候选必须同时满足推理权重 ≤500,000,000 字节和目标环境单图完整流程 ≤100ms；仅有本机 4060/朋友 4090 结果时标为 `LATENCY_UNVERIFIED`。
