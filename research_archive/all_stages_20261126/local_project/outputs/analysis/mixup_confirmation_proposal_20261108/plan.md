# 一个建议方向：EfficientNetV2-S EMA + 固定Mixup的跨折确认

状态：研究方案，尚未实现新调度、未启动训练、未推送GitHub。日期：2026-11-08。

## 假设与来源

假设：固定的图像/标签Mixup能降低模型对训练样本特定模式的依赖，从而改善未参与训练或checkpoint选择的患者上的Study-level Macro-AUC。它是待检验机制，不是已经证实的胸片误差原因。

原Mixup的论文机制依据：https://arxiv.org/abs/1710.09412 。本任务的选择依据主要来自本地真实实验，而不是论文在其他数据集上的收益。

Phase32 Fold1真实记录：
- N17 EMA基准0.8041896584760488。
- P32_NS_MIXUP EMA 0.8089316744872788。
- 差值+0.0047420160112299214；患者Bootstrap CI [0.0012255349181603814, 0.008202369062547217]。
- 最差单类差值仍+0.000673988190734609。
- 这是已被用于选择方向的旧Fold1探索结果，不作为本次确认数据。
- 旧训练日志的早停监控来自859 Study官方Val；新方案因此需要新的同条件BASE。

本地证据文件：
- E:/计挑赛/cxr-multilabel-challenge/outputs/analysis/phase36_release_acceptance/pilot_comparison.csv
- E:/计挑赛/cxr-friend-results/versions/9fd6afc68b904ac4cf7f319053f09023507ca5ee/results/phase32_batch/experiments/P32_NS_MIXUP/resolved_config.yaml
- 同目录train_history.csv、ema_oof_predictions.csv。
- E:/计挑赛/cxr-phase42-dev/src/cxr/phase32/methods.py

## 固定处理与唯一研究因素

两组均使用EfficientNetV2-S、公开ImageNet初始化、384、10类真实标签BCE、Study Mean。
继承原配置：microbatch 8，累积2，有效batch16；AdamW，backbone_lr=1e-4，head_lr=3e-4，weight_decay=1e-4；cosine最长30轮、warmup2；FP16 AMP，clip=1；EMA decay=.999，成功optimizer step后更新，包含buffers；原保守增强不变；主种子42。

BASE：无Mixup。
MIXUP：alpha=.2、probability=1、within_microbatch，lambda~Beta(.2,.2)，图像和完整二值标签使用同一个lambda线性混合。只在训练应用，不搜索alpha。

初始化、批次顺序、微批量、BN、增广、学习率预算、早停规则须成对一致。数据顺序/增强与Mixup使用独立随机流，避免Mixup随机采样意外改变另一组的数据顺序。将这一随机流管理与新验证协议记入配置，不能声称与历史训练完全相同。
运行前审计已有任务防重复；不得用完整Train H17或其他见过外层评估患者的医学任务权重初始化。

## 外层评估与内层早停

保留原冻结患者Fold归属。实现前先读取此前锁定Holdout的患者清单，本轮对照的数据池剔除这些患者；两组执行同样排除并报告实际覆盖数量，不将旧Holdout重新用于模型选择。若无法核验该清单，数据门禁阻塞。针对外层Fold k：
1. Fold k患者完全排除于训练、增广统计、类别支持统计、BN更新和早停。
2. 在其他四折患者内，按SHA256('MIXUP_CONFIRM_V1|'+Subject_id)排序，取前ceil(10%患者)为内层验证，其余为训练；规则与清单先固定，两组相同。此10%是新提出的固定设计值，不是历史最优值。
3. 检查内层每类是否有正负支持；若不足则报告设计阻塞，不反复换seed挑有利划分。
4. EMA的内层Study Macro-AUC用于checkpoint选择，patience6，min_delta0，tie保留较早epoch；不在raw/EMA之间选择。
5. 外层Fold仅在所有训练结束后计算一次最终预测及指标；不得监控外层AUC调epoch。
6. 官方Val与隐藏Test不参与上述训练选择；不在初步运行中产生Test候选。

这里的独立性指当前运行的训练/checkpoint选择与外层评估隔离。Fold2–5患者已出现在过去其他方案的研究中，不能称为全新、从未被查看的盲测集。旧Fold1也不能与本次不同早停协议的结果直接拼成严格同协议完整五折OOF。

## 两阶段，最多8次新训练

第一阶段：Fold2 BASE与MIXUP，共2次。
固定初筛条件：MIXUP−BASE >= .001且无单类点估计下降超过.01。该门槛是拟定的资源决策规则，不是保证泛化的统计定理。报告2000次患者配对Bootstrap、全部类别支持数与差值；即使初筛通过也只能称PROMISING_PILOT。
失败则停止，不换alpha、随机种子或融合比例挽救。

第二阶段：只有初筛通过，才按完全相同配置运行Fold3/4/5的BASE与MIXUP，共6次。不得查看某折结果后改变其余折。
主确认指标只用Fold3–5的拼接折外预测，排除用于新初筛的Fold2及旧筛选Fold1。每个Study只来自没有见过其患者的外层模型。
固定确认标准：合并Macro-AUC差值>=.002、2000次患者Bootstrap95%CI下界>0、至少2/3折点估计正向，并且无单类合并点估计下降超过.01。所有阈值训练前固定，不按实际结果移动。
补充报告Fold2–5合并结果，但不得把含初筛折的结果称为独立确认。
只跑seed42，Bootstrap是对固定训练模型的患者重采样，不覆盖训练随机性；成功仍需保留该限制。

## 怎样通向E08改进

只有确认收益后，才准备全Train的BASE/MIXUP成对训练，使用内层早停轮数的预先固定汇总规则确定训练轮数，官方Val只在训练完成后辅助评估，不回头选择epoch。
以固定E08公式0.5*E03+0.5*新EfficientNet EMA评估组成模型替换；不能同时搜索融合比例。
单模型OOF改善不保证与E03互补性改善，也不保证官方高于77.75。替换前需独立核验预测、来源、部署文件与最终推理入口。
原E08仍为77.75官方基准。E03/H17的77.69仅记录为结果，不用于调本方向参数。

## 成本和部署

历史Mixup训练记录：4090一次10个epoch约2.35465 GPU小时，每轮约14.1分钟，峰值分配显存约2222MiB。它是历史观察，不是新协议耗时保证。
如同吞吐：初始2项约4.7 GPU小时；按30epoch上限线性估算约14.2 GPU小时。数据划分和服务器负载改变会影响实际值，应先短程测速确认预算。
8项总上限按历史吞吐线性估算约56.8 GPU小时，后六项仅在阶段一通过后执行；不能把两阶段算作同时启动的一波8卡任务。
Mixup仅训练时使用，推理仍是单个EfficientNetV2-S，原EMA权重文件约81.66MB，无新增推理模块。不能继承其他候选的T4成绩。

## 继续与停止

当前建议只准备第一阶段两项严格对照；这份文件不构成已经执行的训练结果。
若阶段二未达到确认标准，停止这条固定路线，不凭官方排行榜反馈继续调整。
