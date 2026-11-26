# H50-BASE 与 H17_DISTILL_02 协议差异

H50-BASE 复用 EfficientNetV2-S、384 黑边填充、ImageNet 公开初始化、AdamW 骨干/分类头学习率 1e-4/3e-4、权重衰减 1e-4、余弦调度、EMA 0.999、保守胸片增强与 FP16 AMP。

H17-02 用全部官方 Train、真实标签与教师软标签各 0.5 的组合损失；H50-BASE 只用患者外层 Fold2–5 的真实标签 BCE。外层 Fold1 患者只用于验证及 OOF 评估。H17 已冻结的完整 Train checkpoint 没有用于初始化。

H17 原服务器配置为 micro-batch 8、梯度累积 2。H50-BASE 在本地 8GB GPU 上使用 micro-batch 2、梯度累积 8，有效 batch 均为 16；BatchNorm 的训练批大小因此不同。H50-BASE 只能与该次轨迹的 SWA 作严格配对，不能把历史 H17 的 Val AUC 当作同协议无泄漏对照。

训练最多 4 Epoch；因 patience=6，在正常完成时第 3、4 Epoch 的原始模型 checkpoint 自动用于预先固定的 SWA。SWA 平均完成后，仅用 Fold2–5 训练图重算 BatchNorm 统计量。Fold1 验证指标不会决定 SWA checkpoint 组合。
