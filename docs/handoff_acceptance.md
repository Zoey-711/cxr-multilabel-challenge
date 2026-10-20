# 独立交付验收记录

工作目录为 `cxr-friend-gpu-batch-expanded` 独立 worktree。原项目仍在 RTX 4060 上训练，未修改其代码、配置、数据和 checkpoint。

计划包含原 15 项锁定配置、N17 五折 EMA 与两项 Fold1 Pilot，共 22 项；新增 G/H 四项全 Train 重训/蒸馏槽位为 PENDING，当前不可运行。固定索引：11,055 患者、18,513 Study、30,344 图像，五折患者互斥；患者、Study、图像清单由独立授权附件提供。

合成演练涵盖训练、优化器/调度器/scaler 恢复、纯 state_dict 导出、Val/OOF/Test 概率、基于 ID 的标签关联、固定融合、患者级 Bootstrap、直接结果目录复制、SHA-256 校验、四列全十类 submission 校验和离线复算。CPU 测试明确设置 `CUDA_VISIBLE_DEVICES=-1`；GPU 专项测试跳过，由朋友在获授权设备预检。

本次全量 CPU 验收为 **50 passed、3 skipped，31.29 秒**。合成 22 模型演练验证恢复、立即单模型候选、完整五折 Test 概率、E17 融合、全十类提交 CSV、直接目录复制与 SHA-256 复核；新增 G/H 防误启动及合成教师 OOF 患者隔离检查。冻结索引审计确认患者/Study/图像无跨折泄漏。两个 Pilot 的 torchvision 官方 ImageNet 权重已独立下载并核对完整 SHA-256；没有在正在训练的本机 GPU 上运行 Pilot 前反向，也没有官方目标设备延迟证据。G/H 训练、蒸馏、恢复和完整端到端演练仍为 PENDING，不在本次通过范围内。

执行前阻塞：朋友单机8×RTX4090已确认，但 GPU 小时预算、CPU、RAM、可用存储仍待定；官方完整延迟测试口径待明确。所有部署候选因此保持 `LATENCY_UNVERIFIED` 或其他非 PASS 状态。

官方单图端到端延迟上限为≤100ms；80ms 是内部安全目标，不改变合规边界。模型总权重上限 500,000,000 字节。官方 Val 已多次用于模型选择，所有置信区间仅是探索性证据。内部 Phase15 Holdout 不再用于筛选。

Git 仅用于交付代码；私有数据、ImageNet 权重、checkpoint、凭据和结果文件夹均不得直接推送。GitHub push 尚未获授权。
