# Phase41 本地交付验收

- 队列：BASE/FUSION 各 Fold2～5，准确 8 项；Fold1 不在训练队列。
- 冻结来源：两项 Fold1 的 EMA 权重、OOF、配置、manifest SHA-256 与患者五折索引 SHA-256 已写入 `configs/phase41_plan.yaml`，远程强制复核。
- 条件：BASE/FUSION 同为 micro-batch 4、累积 4、worker 0、有效 batch 16；其余设置来自冻结 Phase39 配置。
- 本地只执行 CPU、合成数据及静态测试；`tests/test_phase41_cpu.py` 与既有 `tests/test_phase35.py` 合计 **24 passed、0 failed、0 skipped**。Phase41 单独 5 passed；Python 文件编译通过。未启动正式训练、未动本地主项目冻结产物。
- 真实 Linux RTX4090 训练、8 卡并发吞吐、显存峰值、共享存储、长时间恢复均待朋友服务器的强制预检及正式运行核实。
- 结果分支：`results/phase41-fivefold`；权重 Release：`phase41-fivefold-v1`。均需朋友配置 `upload.enabled: true` 并授权，默认关闭。

测试运行记录与 Git commit 以实际执行结果为准，不能将远程待验证事项标为 PASS。
