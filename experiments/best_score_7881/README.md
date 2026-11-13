# 历史最高分78.81：E08-EMA + 本地S17-F3

官方成绩由用户确认。Val Macro-AUC=0.8145689350883447。此为历史实验归档，不是新训练队列，不使用排行榜反馈再搜索权重。

## 固定组成

Candidate = 0.75 E08 + 0.25 S17-F3；E08 = 0.5 E03 + 0.5 N01 EMA；E03 = 0.5 E01 + 0.5 M02；E01 = (B01+B02+B03)/3。
最终权重：B01/B02/B03各0.0625，M02为0.1875，N01 EMA为0.375，本地S17-F3为0.25。不能用朋友S17或其他Fold代替。
S17-F3来源为本地Epoch10最佳checkpoint；SHA见candidate_manifest.json。每图输出概率，同Study全部图片Mean。不是二值标签，不是任意单张图像概率。

## 历史生成及复现边界

source/保留原始生成脚本原文及哈希，用于追溯；这些脚本依赖原项目cxr包、其他分析模块及本地冻结目录，不是可独立运行的训练入口。原始S17推理为FP16 batch=8，E08使用冻结Study概率。N01原始Test图片级预测未找到，不能声称六个模型的图片均值均已独立复算；其聚合方式由源码确认。

原项目生成命令（仅在拥有授权数据、完整原项目依赖与匹配哈希的冻结权重时执行）：

```bash
python scripts/generate_e08_s17_f3_candidate.py --archive /path/to/cxr_test_only.zip --output-dir /path/to/NEW_output
```

不得覆盖历史产物。仓库不提供比赛影像、标签、患者索引、Test预测或模型权重；本归档不等于完整部署包。历史CSV保存在本地outputs/submissions/E08_075_S17_F3_025_20261110/submission.csv，SHA见score.json。

## 部署代码快照

deployment/cost.py和predict.py是Study口径V5外层入口，需配套本地最终提交包的model/目录、模型、Test映射及匹配环境才能运行，不能仅靠此Git归档启动。最终包官方命令：

```bash
python cost.py submission.csv pictures_100 pictures_5
```

100个完整Study、5个独立Study预热、batch=1串行、3轮；Study全部图片推理和聚合及终端结果输出计时。只写终端日志。CPU模拟测试通过，未宣称此版本T4速度/历史CSV一致性已通过。历史84.46ms只来自100个完整单图Study的Val参考测试，不代表多图Study新口径成绩。官方最终以复测为准。

## 文件完整性

manifest_sha256.json记录本归档每个文件SHA-256。provenance下原始manifest中的生成时状态（如generated_not_uploaded）保持不改；当前用户确认官方成绩由score.json单独记录。此归档不修改main、历史结果、冻结模型或Submission。
