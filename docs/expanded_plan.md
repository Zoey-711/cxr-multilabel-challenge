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
| PILOT_SELECTED_F02（禁用） | 尚未选择 | 2 | 否 | 条件扩展 |
| PILOT_SELECTED_F03（禁用） | 尚未选择 | 3 | 否 | 条件扩展 |
| PILOT_SELECTED_F04（禁用） | 尚未选择 | 4 | 否 | 条件扩展 |
| PILOT_SELECTED_F05（禁用） | 尚未选择 | 5 | 否 | 条件扩展 |

## 选择与资源限制

尚无冻结的扩展阈值及朋友授权GPU小时，故条件槽位不是可执行任务，22次之后停止。两个Pilot在相同Fold1、相同患者/Study上与D/C/S及固定0.25/0.25/0.5融合比较；边际融合固定0.5参考+0.5Pilot，报告患者Bootstrap及类别退化，不以单折证明稳定优越。

## P1轻量学生方案（设计，不执行）

保留ImageNet DenseNet-121作为轻量学生候选。未来需独立审批固定温度/监督BCE与蒸馏权重，教师只用合法Train数据产物。学生须从合法预训练初始化、单独训练验证并实测完整延迟；不能继承教师AUC。可先复用单模型及小组合评估FP16，不改本批次训练参数。

## 预计成本

A组4060历史粗估42.78GPU小时，按30轮上界粗估105.78小时，不含所有额外评估。N01成功AMP重跑9轮、平均24.29分钟/轮，仅按训练样本比例粗估N17五折14.58本机GPU小时、全30轮48.59小时，固定验证、EMA和Test推理另加。20个五折任务按本机历史粗估至少57.36 GPU小时，两个Pilot未知。朋友目标为单机8×RTX4090，但必须等该机短测速后再估算22次任务GPU小时与墙钟时间；不把4060速度冒充4090实测。模型输出/备份各预留至少50GB，数据另55.23GB；当前预检保守要求同盘至少110GB空闲，不含压缩打包。

## 硬约束

压缩前全部推理权重≤500MB，官方端到端<100ms。只有PASS可部署；完整15/20/22模型均是研究候选，未测速为LATENCY_UNVERIFIED。官方原文≤100ms与用户先前严格<100ms有边界差异，当前采用更严格门槛。
