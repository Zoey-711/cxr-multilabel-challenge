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
| G17_FULL_01（默认关闭） | 五折OOF规则所选第1骨干 | N/A | 继承原骨干 | G组全 Train |
| G17_FULL_02（默认关闭） | 五折OOF规则所选第2骨干 | N/A | 继承原骨干 | G组全 Train |
| H17_DISTILL_01（默认关闭） | densenet121 | N/A | 否 | H组知识蒸馏 |
| H17_DISTILL_02（默认关闭） | efficientnet_v2_s | N/A | 继承N17配置 | H组知识蒸馏 |

## 选择与资源限制

新增 G/H 四项替代旧的 Pilot Fold2–5 占位计划。它们须等待原22项及D/C/S/N完整五折OOF产物核验；两个Pilot仍在基础档。G组只用完整OOF筛选：每骨干五份推理权重各≤500,000,000字节且4090单模型研究p95均<100ms；合格者按OOF Study Macro-AUC降序，平分按D、C、S、N；取前二。全Train Epoch上限取对应骨干五折原始最佳Epoch中位数，其他训练超参数沿用其Fold1配置。4090研究测速不能证明官方硬件合规。

H组教师固定 `0.25×D17_OOF+0.25×C17_OOF+0.50×S17_OOF`；`verify_teacher_sources` 逐Study检查ID、来源折和教师训练患者清单。学生为DenseNet-121、EfficientNetV2-S，固定温度1.0、真实标签BCE与教师软标签BCE各0.5；软标签每图乘以该Study图像数倒数，硬标签维持原图像级权重。H组Epoch上限也取对应骨干五折最佳Epoch中位数。G/H均无独立OOF，输出`N/A`。官方Val用于Epoch选择及重复实验比较，存在选择偏差。真实4090 G/H训练预检仍待朋友服务器验证。

默认只执行22项。用户确认、朋友另外授权正数G/H GPU小时预算后，才在忽略的环境配置中设置`CXR_ENABLE_GH=1`、`CXR_GH_GPU_HOURS`，同一入口再执行四项；`CXR_GPU_HOURS`控制两阶段合计。依赖、选择、教师隔离、额外预算或真实GPU预检失败均阻止扩展启动，绝不补齐不存在的指标。四项各自保存Val/Test概率、完整配置、SHA、恢复checkpoint、研究性4090延迟及全十类Submission候选；固定`G17_PAIR`、`H17_PAIR`等权精简融合不得依据隐藏Test调权。

两个Pilot在相同Fold1、相同患者/Study上与D/C/S及固定0.25/0.25/0.5融合比较；边际融合固定0.5参考+0.5Pilot，报告患者Bootstrap及类别退化，不以单折证明稳定优越。G/H 完成后才可以新增其 Val/Test 概率、全十类 Submission 候选；严格保留≤500MB与目标设备端到端<100ms双约束。未在目标设备实测则为 `LATENCY_UNVERIFIED`。

## H组学生与部署解释

DenseNet-121和EfficientNetV2-S均重新从已核验的ImageNet权重初始化。蒸馏后的学生必须单独计算官方Val AUC、真实文件大小与完整推理时间，不能继承教师集成AUC。固定`G17_PAIR`与`H17_PAIR`的权重各0.5，只有在官方目标设备测得完整流水线<100ms且全部权重≤500MB时才可标PASS；4090研究测速不足以证明T4合规。

## 预计成本

A组4060历史粗估42.78GPU小时，按30轮上界粗估105.78小时，不含所有额外评估。N01成功AMP重跑9轮、平均24.29分钟/轮，仅按训练样本比例粗估N17五折14.58本机GPU小时、全30轮48.59小时，固定验证、EMA和Test推理另加。20个五折任务按本机历史粗估至少57.36 GPU小时，两个Pilot未知。朋友目标为单机8×RTX4090，但必须等该机短测速后再估算22次任务GPU小时与墙钟时间；不把4060速度冒充4090实测。模型输出/备份各预留至少50GB，数据另55.23GB；当前预检保守要求同盘至少110GB空闲，不含压缩打包。

G/H 四项训练预算在原22项完成后用朋友4090的实测五折训练记录估计：对应骨干每Epoch耗时中位数 × `30344 / 当前Fold训练图像数` × 五折最佳Epoch中位数；H组再加15%软标签损失开销的**粗估**。训练后独立Val/Test/延迟评价尚未包含。四项合计小时数写入冻结的扩展选择清单；实际启动仍以朋友授权的总GPU小时与G/H子预算两道闸门约束，不能把估计代替授权。当前未取得4090实测吞吐，不能给出可信固定GPU小时或墙钟数。

## 硬约束

压缩前全部推理权重≤500MB，官方端到端<100ms。只有PASS可部署；完整15/20/22模型均是研究候选，未测速为LATENCY_UNVERIFIED。官方原文≤100ms与用户先前严格<100ms有边界差异，当前采用更严格门槛。
