# Phase33：H17-02 分类头微调备份实验

本目录与冻结 H17-02、E08、Phase17 和朋友的两批新队列隔离。唯一实验为 `P33_H17_HEAD_01`：读取经 SHA-256 核验的 H17-02 EfficientNetV2-S 原始推理权重，复制到本实验目录；冻结骨干及 BatchNorm 缓冲区，只更新 10 类线性分类头。输入为 384×384 `black_pad_v1`，使用原 H17 的保守训练增强、完整官方 Train、0.5 真实标签 BCE + 0.5 患者隔离教师 OOF BCE，软标签按每个 Study 的图像数倒数加权。AdamW 分类头学习率固定 `1e-5`，微批 8、梯度累积 2、FP16 AMP，最多 5 Epoch，Val Study Macro-AUC 连续两轮不刷新则停止。没有学习率搜索，也没有 Test 标签或 Submission。

原始 H17-02 使用完整 Train 训练 4 Epoch，原训练配置含 AdamW backbone `1e-4`、head `3e-4` 和 EMA。此次只有纯推理权重，没有原优化器状态，故这是**新优化器的分类头微调**，不是原训练的精确断点续训。学生预测训练集不构成独立 OOF；859 Study 官方 Val 已反复用于开发，比较只能标记 EXPLORATORY。

`phase33_head_pilot.py --smoke` 的教师覆盖、患者来源、损失、单步 CUDA 和保存恢复测试已通过。Windows 当前环境多进程 DataLoader 创建管道时报 WinError 5，本实验固定 `num_workers=0`，未改动图像转换、增强、损失或模型参数。`P33_H17_HEAD_01/` 中每轮保存权重与完整新优化器恢复状态。正式训练完毕后，`watch_phase33.py` 只调用 CPU 脚本 `finalize_phase33.py`，生成逐类指标、患者级 Bootstrap、模型大小及最终决策；未完成前这些文件不存在，不应引用虚构结果。

查看状态：`P33_H17_HEAD_01/train_history.csv`、`P33_H17_HEAD_01/status.json`、`pilot_stdout.log` 和 `pilot_stderr.log`。如进程异常退出，先核对最后一份 `last_training.pt`；只有状态未完成且配置一致时，才能人工运行 `phase33_head_pilot.py --resume`。
