# L01: ASL 损失函数受控实验计划（尚未启动）

## 对照定义

控制组是冻结的 M02 **实际运行配置**。L01 使用同一 ConvNeXt-Tiny ImageNet 权重、训练/验证数据、384×384 黑色 Padding 预处理、保守增强、普通图像级 shuffle 采样、batch 8、累积 2、AdamW、两组学习率、权重衰减、cosine + 2 warmup epochs、FP16 AMP、种子 42、Study Mean 概率聚合及 Macro-AUC 最大化选择 checkpoint。唯一训练算法变化为 `BCEWithLogitsLoss` → ASL；实验名和 ASL 参数属于配置记录。最多 30 epoch，早停 patience 6。输出目录使用 `outputs/experiments/l01_convnext_tiny_asl_384`，必须不存在才可启动。

ASL 固定预设参数：`gamma_neg=4`, `gamma_pos=1`, `clip=0.05`, `eps=1e-8`, `detach_focal_weight=true`，对应官方参考实现常用默认值。ASL 对容易的阴性施加较小权重，并用概率裁剪限制极端阴性；在多标签不平衡任务中可能有益，但此处不做参数搜索。训练循环在 FP16 autocast 中把 ASL 数学运算转为 float32，并延续 M02 现有的有效标签掩码及按实际有效标签数归一化；梯度累积、优化器更新与调度器逻辑保持原样。

方法依据：[ASL 原论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf)、[作者公开实现](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)。

## 启动前检查结果

`scripts/preflight_asl.py --config configs/l01_convnext_tiny_asl_384.yaml` 已通过：配置递归差异只包含实验名、loss 类型与 ASL 参数；缓存的 torchvision ImageNet 权重 SHA-256 与 M02 的 `pretrained_source.json` 一致；真实 Train/Val 各读取一张，形状均为 `3×384×384`，标签 10 维；CUDA 上合成 logits 的 ASL 前反向均有限。完整证据在 `outputs/analysis/phase9_asl_preflight_20261017.json`。这不是正式训练，也没有加载完整模型或执行优化器更新。

## 将来获批后的评价

按 Study-level Macro-AUC 选最佳模型，独立复算全部 859 个 Study 的 10 类 AUC、AP 和 Macro-F1@0.5；与冻结 M02 做 228 名患者的配对 bootstrap，报告 Macro-AUC 差值 95% 区间及类别 1、4、8 变化。查看训练/验证曲线、早停与显存；负结果同样保留。不得用隐藏测试标签或排行榜结果挑选 ASL 参数。

本阶段不启动 L01；后续启动须由用户明确确认。
