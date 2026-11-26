# Phase50 Local B 运行中

H50-BASE Fold1 于 2026-10-30 14:13:47 启动。训练 PID 61924（实际计算子进程 PID 4160），收尾监控 PID 19536。

真实 Train 图像短测：16 张图、1 次有效优化器更新、AMP 跳步 0、损失 0.691926、checkpoint 恢复输出最大差 0、峰值分配显存 735.25 MiB。

最近核验：第 1 Epoch 100/1521 次更新，AMP 跳步 0，峰值分配显存 892.99 MiB。正式 Fold1 OOF AUC、最佳 Epoch、SWA AUC 尚未产生。请读取 `H50_BASE_F1/status.json`、`H50_BASE_F1/train.log` 和 `H50_BASE_F1/launcher.stderr.log` 查看当前状态。每轮结束保存 `last_training.pt`；训练正常完成后自动运行固定 Epoch 3/4 的 SWA 收尾，结果写入 `H50_SWA_F1/` 与 `final_decision.md`。

KD 保持 BLOCKED；内层教师来源要求见 `kd_teacher_rebuild.md`。
