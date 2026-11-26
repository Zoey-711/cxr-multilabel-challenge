# 最终提交材料整改清单

依据：https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf 第 7、8、10 条及附录 A。此处只审计，不生成或提交最终材料包。

| 材料/条件 | E08 | H17-02 |
|---|---|---|
| 历史最高成绩 `submission.csv` | 已有冻结 E08 文件及哈希；本轮未改动 | 已有朋友候选 CSV，尚非当前最高成绩 |
| 最终推理纯权重与 SHA | 5 份均 PASS | 单个纯权重已下载并通过本地 SHA 校验 |
| 必要配置、类别表及 500MB 计数 | 已核配置哈希和 278,442,617 字节；最终材料目录仍需重算 | 本地权重加现有配置 81,654,329 字节；最终材料目录仍需重算 |
| `predict.py`/`cost.py` 同一模型入口 | Phase29 独立副本已实现；未合入正式路径 | 同一入口已完成 859 Val 和本地测速 |
| 官方 5+100×3、T4 16GB、batch1 | PENDING | PENDING |
| cost 结果与最终 submission 一致 | 本地单图 Val 最大差 0.000793 > 1e-4，尚未通过 | 本地单图 Val 最大差 0.000732 > 1e-4，尚未通过 |
| `train.py`、`model.py`、`utils.py`、`requirements.txt` 可复现代码 | 原项目有相关源码；材料目录尚未组装/验收 | 需取得朋友训练代码对应 commit |
| 原始预训练权重及 SHA | 训练记录有来源哈希，本地原始文件待定位 | 同左 |
| 许可证全文、上游训练数据 model card/论文 | 已取得 torchvision 软件 BSD-3 原文；逐权重再分发许可/说明待确认 | 同左 |
| 模型核验材料与数据申明清单.xlsx | PENDING | PENDING |
| 朋友使用比赛数据的授权与数据声明 | 不适用本地训练 | MANUAL_CONFIRMATION_REQUIRED |
| 最终提交材料包与主办方复测 | PENDING | PENDING |

官方 PDF 要求材料包中含 `code/`、`model/`、`pretrain_model/`、`licenses/`、`cost.py`、核验 Excel 和历史最高分 `submission.csv`。现有 Phase29 目录是审计副本，**不是已验收的最终材料包**。
