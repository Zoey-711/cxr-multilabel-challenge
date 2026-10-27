# 历史实验说明与代码索引

本文件用于阅读已上传的本地实验代码和朋友批量实验协议。完整的本地代码、配置、研究文档目录在 [`research_archive/local_20261026/`](../research_archive/local_20261026/)；[归档清单](../research_archive/local_20261026/archive_manifest.json)记录每个源文件及脱敏后文件的 SHA-256。该归档是研究快照，**不会因克隆仓库而启动训练**；原始影像、标签、预训练权重、checkpoint、预测 CSV 和训练日志均未作为本次文档上传。

| 实验阶段 | 研究问题和对照 | 代码或说明入口 | 证据边界 |
|---|---|---|---|
| B01–T01、E01–E08 | DenseNet、ConvNeXt、Swin、EfficientNetV2-S 的基线、增强、EMA 和固定融合 | [历史 README](../research_archive/local_20261026/README.md)、[历史配置](../research_archive/local_20261026/configs/) | 历史官方 Val 被反复使用；E08 为冻结提交基准，不能以 Val 微小差异推断独立泛化改善。 |
| Phase17 | 相同患者级五折上的 D17/C17/S17；每个 Study 只由未训练其患者的 Fold 产生 OOF | [五折协议](../research_archive/local_20261026/docs/phase17_joint_folds.md)、[朋友 26 项说明](../README_friend.md) | 本地 15 模型五折 OOF 为 18,513 Study、11,055 患者；完整教师集成默认只作研究，不自动满足部署限制。 |
| Phase18–20 | 弱类别、患者级误差、轻量候选和 E08 推理流程诊断 | [历史分析脚本](../research_archive/local_20261026/scripts/)、[历史 README](../research_archive/local_20261026/README.md) | 不访问隐藏 Test 标签；本地 4060 时延不是官方目标硬件证明。 |
| Phase25 | 用合法图像级 OOF 比较 Study Mean、Max、固定混合与视角规则；审计标签语义 | [聚合审计](experiment_notes/phase25_study_aggregation_audit.md)、[标签审计](experiment_notes/phase25_label_semantics_audit.md)、[代码](../research_archive/local_20261026/src/cxr/engine/phase25_audit.py) | Mean OOF .803049，Max .797897，半 Mean/Max .801338；不因微小差异搜索更多规则。 |
| Phase26 | 完整 OOF 教师软标签的本地 DenseNet 蒸馏备份 | [历史代码及配置](../research_archive/local_20261026/) | 全 Train 学生无独立 OOF；不能将其训练集预测称为 OOF。 |
| 朋友 Phase17-Batch | 26 项 D/C/S/N 五折、Pilot、G 全 Train 和 H 蒸馏 | [README_friend](../README_friend.md)、[26 项验收](phase17_26_acceptance.md) | 朋友结果独立于本地冻结实验；G/H 没有独立 OOF。 |
| Phase32、E08-Tune | 8 项单变量 Pilot；围绕 N01/N17 EMA 的固定参数对照 | [Phase32](phase32/README.md)、[E08-Tune](e08_tune/README.md) | Fold1 选择效应须由后续折验证；不能用隐藏榜单选择参数。 |
| Phase29–31 | E08/H17 的预测一致性、完整推理链和 T4 部署材料准备 | [Phase29](experiment_notes/phase29_deployment_decision.md)、[Phase30](experiment_notes/phase30_deployment_report.md)、[Phase31](experiment_notes/phase31_h17_deployment_report.md) | 4060 测速仅为本地参考；无官方 T4 实测时维持 `T4_UNVERIFIED`。 |
| Phase33 | H17-02 冻结骨干后的分类头微调备份 | [实验协议快照](experiment_notes/phase33_h17_head_finetune.md) | 全 Train 初始化权重不适合作为 Fold1 无泄漏 OOF 对照。 |
| Phase34 | DenseNet 训练期辅助头与同条件 BASE 对照 | [本次收录的结论](experiment_notes/phase34_final_decision.md) | Fold1 差值 −.000982，患者 Bootstrap 区间跨 0；已冻结为无明确收益。 |
| Phase35 | 复用 Phase32 已完成 Fold1，补类别加权 BCE 384 和 BCE 512 各四折 | [Phase35](phase35/README.md) | 完整 OOF 分别 .809691 / .809310；排除筛选用 Fold1 后收益不稳定。 |
| Phase38 | 对弱类别和特征表示的研究设计 | [研究方案快照](experiment_notes/phase38_research_plan.md) | 仅设计；不是实验结果或病灶定位证据。 |
| Phase39 | 同 Fold1 的 EfficientNetV2-S BASE 与保留中层特征的 FUSION 成对实验 | [结构](phase39_model_architecture.md)、[代码](phase39_local_code.md)、[结论](experiment_notes/phase39_final_decision.md) | BASE .804757、FUSION .807709；探索性单折，不能冒充完整五折确认。 |

各行结果只是帮助定位文件的摘要，复算应回到对应的冻结预测和 manifest。GitHub 代码分支不含原始比赛数据，也不含上述训练结果的逐 Study 概率。需要对照比赛限制时，以[2026-10-23 官方规则原件](competition/README.md)为准，不能以归档中的旧规则摘要为准。
