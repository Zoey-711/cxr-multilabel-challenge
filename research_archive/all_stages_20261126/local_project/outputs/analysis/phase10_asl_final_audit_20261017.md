# Phase 10 L01 启动前最终审计

**结论：通过。** 原配置未改动；正式 L01 使用 `configs/l01_convnext_tiny_asl_384.yaml`。完整数值记录在同名 JSON。

## 公式

逐元素实现与 [ASL 作者公开实现](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)核对：`gamma_neg=4`, `gamma_pos=1`, `clip=0.05`, `eps=1e-8`，负类概率先加 margin 后上限裁剪，正/负类分别取 log，然后乘以 detached 的非对称 focal 权重。固定样例上损失及 logits 梯度最大绝对误差均为 **0**。ASL 内部将 logits 转 float32，并对概率做 `eps` 下限裁剪，FP16 autocast 下前反向结果有限。

## Reduction、掩码与累积

ASL 返回 `[micro_batch, 10]` 未归约损失；训练循环使用 `(per_label_loss × valid_mask).sum() / group_valid`。`group_valid` 来自同一梯度累积组内全部 microbatch 的有效标签总数，包括最后不足 2 个 microbatch 的组。每组只进行一次 `optimizer.step`/`scheduler.step`；若 GradScaler 跳步则不推进 scheduler。训练集 30,344 张图、batch 8、累积 2，对应每 epoch 3,793 个 microbatch 和 1,897 个累积组，末组只有 1 个 microbatch。单元测试覆盖了掩码及最后组归一化。

## 同一初始 forward 的损失与梯度

用同一组 ImageNet 初始化的 ConvNeXt-Tiny 参数、相同的两张真实训练图像及同一次初始 forward，分别反向传播 BCE 与 ASL，**不执行优化器更新**：

| 指标 | BCE | ASL | ASL/BCE |
|---|---:|---:|---:|
| 平均有效标签损失 | 0.669481 | 0.063370 | 0.0947 |
| 全模型梯度范数 | 5.657511 | 0.791394 | 0.1399 |
| 骨干梯度范数 | 5.284073 | 0.717926 | 0.1359 |
| 分类头梯度范数 | 2.021387 | 0.332997 | 0.1647 |

数值均有限且非零。ASL 的较小绝对值符合其下调易阴性项的预设公式；不同损失的绝对标量不能直接当作收敛优劣。该比较只用两张图像，是启动前数值检查，不代表全训练集梯度分布。

## 来源与隔离

预训练权重是与 M02 相同的 torchvision ImageNet 文件，SHA-256 为 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`。训练数据仍是主办方 ZIP，未接入外部标注或医学预训练权重。L01 与 M02 的递归配置差异仅为实验名、损失名及 ASL 固定参数；随机种子、采样、增强、验证协议和优化器设置一致。L01 使用独立输出目录，不写入已冻结实验。
