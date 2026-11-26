# Phase50-Local 预检与执行决定

本轮正式训练未启动。预检报告见 `preflight_report.json`；CPU 与 RTX4060 短程合成数据训练各成功完成 1 次 optimizer 更新，loss、logits 和梯度有限，EMA 更新成功，CUDA 峰值分配显存 657.5 MiB。训练集 ZIP 中 30,344 张图像的索引可读，并实际解码和按 384×384 原协议预处理一张训练图像。本机预检时 4060 可用显存约 6,657 MiB，无已识别的 Python 计算进程。这些检查只能证明短步可运行，不能证明多轮训练在 8GB 显存中持续稳定。

原 H17_DISTILL_02 使用全部 11,055 名 Train 患者、18,513 Study、30,344 张图像，在 4 Epoch 内以 ImageNet EfficientNetV2-S 初始化，384 输入，AdamW（骨干 1e-4、分类头 3e-4）、weight decay 1e-4、余弦调度与 2 Epoch warmup、EMA 0.999、FP16 AMP、micro-batch 8×累积 2，并使用 0.5 真实标签 BCE + 0.5 Study 权重教师 BCE。其最佳推理权重不是 Fold1 无泄漏初始化，原模型没有独立 Fold1 OOF。改变为 micro-batch 2×累积 8 会改变 BatchNorm 统计，因此新 BASE 和 KD 必须从同一 ImageNet 权重以相同微批和设置重新训练；历史 H17 Val 0.808610 只能作为背景参考。

**阻断原因：现有教师 OOF 不能用于本轮 Fold1 学生验证协议。** 它对每位被预测患者自身是折外的：18,513 Study 的来源索引、患者 Fold 和 D17/C17/S17 模型 ID 均核对通过。可是用 Fold2–5 患者作为学生训练集时，教师预测分别来自相应 Fold2–5 模型；这些教师模型在训练时包含了 Fold1 的 2,211 名学生验证患者。因此教师软标签已经间接依赖学生验证集，不能据此做无泄漏的 H50-KD 与 H50-BASE Fold1 配对比较。满足当前要求需重新建立外层 Fold1 隔离的内层教师，或取得明确未见 Fold1 患者的教师预测；现有五折 OOF 不够。

| 实验 | 当前状态 | OOF AUC | 原因 |
| --- | --- | --- | --- |
| H50-BASE | NOT_STARTED | N/A | 为保持成对实验意义，教师隔离门禁未通过时不启动正式训练；不存在可复用的同协议 Fold1 H17 基线。 |
| H50-KD | BLOCKED | N/A | 现有教师对学生 Fold1 验证集存在外层隔离问题。 |
| H50-SWA | BLOCKED | N/A | 没有 H50 同一次训练轨迹的多个兼容 Epoch checkpoint；只有一个冻结 H17 推理权重不能计算 SWA，BatchNorm 统计还需同数据重新估计。 |

此前 Phase33 已对全 Train H17 做过分类头微调，Val 无增益；它是全 Train 训练内实验，不能充当本轮 Fold1 基线。Phase41R 修复记录说明 AMP 溢出可由 GradScaler 显式跳步并计数，连续事件超过阈值才失败；本轮 Smoke Test 遵循 unscale→clip→scaler.step→scaler.update 顺序，0 跳步。完整训练实现若继续，必须记录每 Epoch 成功更新、AMP 跳步与显存峰值，并在跳步异常增多时停止。

当前无 H50 OOF、Bootstrap 或部署候选，不能评价 AUC 改善。建议先解决外层教师隔离问题，再以固定 Fold 和同条件 BASE/KD 开展训练。冻结 E08、H17、Phase17 和既有 Submission 均未修改。
