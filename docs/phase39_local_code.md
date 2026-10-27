# Phase39 本地实验代码快照

本分支收录用户在本地RTX4060上运行的Phase39单折配对实验代码。实验为
`P39_BASE_F1` 和 `P39_FUSION_F1`：同一冻结Fold1、EfficientNetV2-S ImageNet
初始化、384输入、BCE、EMA 0.999和Study Mean。FUSION把`features[5]`
的全局池化向量投影至64维，与末端1280维向量拼接后生成最终10类logits。
推理时必须保留融合分支。

这些文件只是**代码与配置**；本地训练产生的checkpoint、概率和日志没有提交。
Phase39不属于原26项、Phase32、E08-Tune或Phase35任务队列，克隆本分支不会
自动启动Phase39。本地配对实验已经结束：BASE Fold1 OOF Macro-AUC 为
0.804757，FUSION 为 0.807709，差值 +0.002951；患者 Bootstrap 95% CI
[+0.000190, +0.005699]。这是历史多次使用的 Fold1 上的探索结果，
不代表完整五折确认。原始权重和逐 Study 预测未上传本代码分支，结论快照见
[`experiment_notes/phase39_final_decision.md`](experiment_notes/phase39_final_decision.md)。

源码：`src/cxr/phase39_model.py`、`src/cxr/phase39_run.py`、
`src/cxr/phase39_support.py`、`scripts/run_phase39.py`；冻结配置：
`configs/phase39/base_config.yaml`及`fusion_config.yaml`。
患者折索引由仓库已有`outputs/analysis/phase17_patient_folds/`提供。

在**独立、授权**的环境复现时先设置环境变量：

```bash
export CXR_PHASE39_ARCHIVE=/authorized/data/cxr_first.zip
export CXR_PRETRAINED_ROOT=/authorized/imagenet-cache
python scripts/run_phase39.py --action audit
python scripts/run_phase39.py --action preflight
python scripts/run_phase39.py --action run
```

预训练根目录应含`hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth`。
脚本按配置哈希核对Fold索引和ImageNet权重。`run`顺序运行BASE与FUSION，
重复执行会核验已完成任务并从最近完整Epoch恢复未完成任务。
结果写入被Git忽略的`outputs/experiments/phase39_multistage_fusion/`。
状态查看：`python scripts/run_phase39.py --action status`。

本地Windows训练使用单进程DataLoader，是因为命名管道权限限制；
两组处理一致，但与历史朋友N17的worker数不同。本目录为探索性Fold1结果，
不得称作完整五折OOF或官方T4合规证明。
