# Phase17 扩展计划（22+最多4）

原15任务哈希锁定；以下顺序由依赖与授权GPU决定，不代表GPU并发已获批准。

| ID | 骨干 | Fold | EMA | 优先级 |
|---|---|---:|---|---|
| D17_F01 | densenet121 | 1 | False | P0 |
| D17_F02 | densenet121 | 2 | False | P0 |
| D17_F03 | densenet121 | 3 | False | P0 |
| D17_F04 | densenet121 | 4 | False | P0 |
| D17_F05 | densenet121 | 5 | False | P0 |
| C17_F01 | convnext_tiny | 1 | True | P0 |
| C17_F02 | convnext_tiny | 2 | True | P0 |
| C17_F03 | convnext_tiny | 3 | True | P0 |
| C17_F04 | convnext_tiny | 4 | True | P0 |
| C17_F05 | convnext_tiny | 5 | True | P0 |
| S17_F01 | swin_t | 1 | False | P0 |
| S17_F02 | swin_t | 2 | False | P0 |
| S17_F03 | swin_t | 3 | False | P0 |
| S17_F04 | swin_t | 4 | False | P0 |
| S17_F05 | swin_t | 5 | False | P0 |
| N17_F01 | efficientnet_v2_s | 1 | True | P0 |
| N17_F02 | efficientnet_v2_s | 2 | True | P0 |
| N17_F03 | efficientnet_v2_s | 3 | True | P0 |
| N17_F04 | efficientnet_v2_s | 4 | True | P0 |
| N17_F05 | efficientnet_v2_s | 5 | True | P0 |
| CS17_F01 | convnext_small | 1 | False | P2 |
| NM17_F01 | efficientnet_v2_m | 1 | False | P2 |
| G17_FULL_01（PENDING） | 待预定规则从完整 OOF 中选择 | N/A | 否 | G组全 Train |
| G17_FULL_02（PENDING） | 待预定规则从完整 OOF 中选择 | N/A | 否 | G组全 Train |
| H17_DISTILL_01（PENDING） | 待冻结学生和蒸馏参数 | N/A | 否 | H组知识蒸馏 |
| H17_DISTILL_02（PENDING） | 待冻结学生和蒸馏参数 | N/A | 否 | H组知识蒸馏 |

## 选择与资源限制

新增 G/H 四项替代旧的 Pilot Fold2–5 占位计划。它们都必须等待 D/C/S/N 各五折共20项训练、完整 OOF 与评价完成；两个 Pilot 仍是基础档的一部分，22项须优先执行。G组只可依照**训练前冻结**的完整 OOF 和部署筛选规则确定两个骨干，不能按隐藏 Test 或重复使用的官方 Val 追逐分数。H组教师必须使用每位患者所属 Fold 的未见患者模型 OOF；`cxr.batch.extension.verify_teacher_sources` 会逐 Study 核查 ID、来源模型、所属折和各模型训练患者清单。仅这一核查并不构成完整蒸馏实现。

目前 G组选择阈值、训练轮次和两个骨干尚未冻结；H组教师融合公式、温度、真实标签/软标签损失权重、学生模型与训练恢复路径尚未实现并验证；朋友新增 GPU 小时也尚未授权。因此四项均为 `PENDING`、不可运行，`--allow-extended` 明确报错，22项之后停止。未来必须用独立端到端测试、人工确认计划与授权预算后才可启用。G/H 均为全 Train 训练，没有独立 OOF，输出须写 `N/A`，不能伪造。

两个Pilot在相同Fold1、相同患者/Study上与D/C/S及固定0.25/0.25/0.5融合比较；边际融合固定0.5参考+0.5Pilot，报告患者Bootstrap及类别退化，不以单折证明稳定优越。G/H 完成后才可以新增其 Val/Test 概率、全十类 Submission 候选；严格保留≤500MB与目标设备端到端<100ms双约束。未在目标设备实测则为 `LATENCY_UNVERIFIED`。

## P1轻量学生方案（设计，不执行）

保留ImageNet DenseNet-121作为轻量学生候选。未来需独立审批固定温度/监督BCE与蒸馏权重，教师只用合法Train数据产物。学生须从合法预训练初始化、单独训练验证并实测完整延迟；不能继承教师AUC。可先复用单模型及小组合评估FP16，不改本批次训练参数。

## 预计成本

A组4060历史粗估42.78GPU小时，按30轮上界粗估105.78小时，不含所有额外评估。N01成功AMP重跑9轮、平均24.29分钟/轮，仅按训练样本比例粗估N17五折14.58本机GPU小时、全30轮48.59小时，固定验证、EMA和Test推理另加。20个五折任务按本机历史粗估至少57.36 GPU小时，两个Pilot未知。朋友目标为单机8×RTX4090，但必须等该机短测速后再估算22次任务GPU小时与墙钟时间；不把4060速度冒充4090实测。模型输出/备份各预留至少50GB，数据另55.23GB；当前预检保守要求同盘至少110GB空闲，不含压缩打包。

## 硬约束

压缩前全部推理权重≤500MB，官方端到端<100ms。只有PASS可部署；完整15/20/22模型均是研究候选，未测速为LATENCY_UNVERIFIED。官方原文≤100ms与用户先前严格<100ms有边界差异，当前采用更严格门槛。
