# Phase17 本地五折结果交接

15/15 个最佳 checkpoint、运行配置、Val/OOF 概率和冻结 SHA-256 均已核验。
18,513 个 Train Study / 11,055 名患者各有且仅有一个所属 Fold 的 OOF 来源，标签与10类顺序一致。
完整 E17 OOF AUC 0.803049；五折平均 Val AUC 0.808862。
统一概率文件为 e17_oof_study_probabilities.csv；对应 Fold 和三模型版本见 e17_oof_source_index.csv。
朋友计划的折划分、D17/S17 原训练配置匹配；C17 额外维护 EMA，普通权重需与 EMA 明确区分。
朋友模型独立训练，不能仅凭配置相同就混用其 OOF；必须先通过 import_phase17_friend_results.py 的 manifest/标签/索引检查。
朋友 N17 EMA 正式结果未同步：PENDING；不得填写对比数值。
官方 Val 曾重复用于早停和模型选择，Val 差异不能当作独立泛化证据。
完整15模型超过500MB，且目标硬件延迟未验证；仅用于研究，不生成 Submission。
