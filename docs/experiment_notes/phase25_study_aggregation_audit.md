# Phase25 Study 聚合审计

来源：官方 Train 原始 ZIP 的 `train.csv`；冻结的 Phase17 五折索引；D17/C17/S17 各五折的 `oof_analysis/image_predictions.csv`、`study_predictions.csv` 和 `frozen_manifest.json`。15份图像预测共覆盖每骨干30,344张图像，逐一核对患者、Study、图像ID、所属Fold及文件SHA-256。由逐图概率重建的三骨干 Mean 与冻结 Study 预测完全一致，最大绝对误差为0；固定E17 Mean与冻结完整OOF按ID对齐后也完全一致。

所有Train患者只由未见其训练Fold的模型给出OOF。用于本表的固定融合为 `0.25 D17 + 0.25 C17 + 0.50 S17`，三个骨干各在相同Study上先独立聚合，再融合。这里没有使用训练拟合内的概率。单图Study 8,507个，多图10,006个；视角来自原始CSV的 `ViewPosition`：AP 12,152、PA 7,604、LATERAL 6,554、LL 2,831、空白1,202、`PA LLD` 1。原始字段可恢复图像与Study/患者/视角的映射。

在读取结果前固定的规则：Mean；Max；`0.5 Mean + 0.5 Max`；ViewBalanced（AP/PA为正位组，LATERAL/LL为侧位组，先组内均值，再将存在的组等权平均）。含空白或未识别视角的Study回退整个Study Mean，共728个Study。单图四种规则完全相同。ViewBalanced仅改变1,307个Study，所有10类概率的平均绝对变化0.000611；没有从标签拟合视角权重。

| 规则 | 完整OOF Macro-AUC | 相对Mean | 11,055患者配对bootstrap 95% CI | 多图Study AUC | 官方Val五折平均AUC |
|---|---:|---:|---:|---:|---:|
| Mean | 0.803049 | 0 | 基准 | 0.835676 | 0.808862 |
| Max | 0.797897 | -0.005152 | [-0.005933, -0.004403] | 0.832429 | 0.806498 |
| 0.5Mean+0.5Max | 0.801338 | -0.001711 | [-0.002143, -0.001304] | 0.834660 | 0.808581 |
| ViewBalanced | 0.803201 | +0.000152 | [+0.000030, +0.000277] | 0.835975 | 0.809138 |

Bootstrap固定seed=42，完整集合和多图集合各有2,000/2,000次有效患者抽样，0次无效；同一患者所有Study共同重采样，任一类别缺正/负例则整次抽样无效。ViewBalanced在五个OOF Fold中的AUC增量均为正，约+0.00008至+0.00025；多图增量+0.000299，95% CI [+0.000047,+0.000551]。这些区间条件于已用官方Val选择过Epoch的现有模型，并且本轮检查了多个固定规则，不能当作全新独立验证。Val也已反复用于模型选择。

结论：Max和半Mean半Max均稳定降低AUC。ViewBalanced有方向一致但极小的改善，尚不足以证明实际部署收益，尤其不值得为此训练新的聚合头。此前M01 Val实验中Max和半Mean半Max低于Mean；V00/V01/V02/V03的学习式或残差头也未给出可靠的进一步复杂化收益。完整数值见 `aggregation_oof_comparison.csv`（全部10类、单图/多图和各骨干）、`aggregation_fold_comparison.csv`、`aggregation_val_check.csv`、`bootstrap_comparisons.json`。官方Val为探索性一致性检查，不用于重新选融合权重。
