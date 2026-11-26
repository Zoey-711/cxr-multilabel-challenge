# Phase25 结果

全部15份Phase17图像级OOF均存在并通过SHA、患者/Study/图像/Fold检查，三骨干的Study Mean重建误差为0。完整E17 OOF Mean 0.803049；Max 0.797897；半Mean半Max 0.801338；固定视角组等权0.803201。官方Val同规则五折平均分别为0.808862、0.806498、0.808581、0.809138。视角规则改善很小，仍需新患者证据；没有启动训练。

原始标签与当前十类编码一致，未观察到空标签、独立不确定字段、多图Study标签冲突或No Finding与疾病标签同时阳性。类别8的低AUC不能由这些检查归因为编码错误。下一项优先投入是收集真正独立且含逐图概率的结果，验证固定视角规则；当前没有值得启动的4060重训练Pilot。

分析代码：`src/cxr/engine/phase25_audit.py`、`phase25_val_check.py`。配置：`configs/phase25_aggregation_audit.yaml`。定量表：`aggregation_oof_comparison.csv`、`aggregation_fold_comparison.csv`、`aggregation_val_check.csv`、`bootstrap_comparisons.json`；来源哈希和完整标签统计：`source_audit.json`、`label_audit.json`。本目录是新生成的分析副本，冻结checkpoint、原始OOF及E08产物均未改动。
