# Phase 17-Batch GitHub 发布前检查（2026-10-20）

## 本地代码状态

- 分支：`friend-gpu-batch`。本次检查开始时提交为 `4f1d9176369e14671f75e61df6a293e1f1483dda`，工作区干净。
- Git remote：**未配置**，因此目前无法核对目标仓库地址和私有属性，也未执行 push。
- 计划静态校验：22 项基础任务 + 4 项自动依赖的 G/H 任务，`default_tasks=26`。
- 正式启动必须持有同一主机、同一计划哈希、覆盖授权 GPU 的 `REMOTE_PREFLIGHT` 通过凭证，且须填写总 GPU 小时与 G/H 子预算。
- G/H 先核对 20 个五折任务、完整患者隔离的 OOF 教师来源，再用各自正式配置运行一个真实 AMP 优化器步。短测会保存并恢复模型、优化器、调度器、GradScaler 与随机数状态，对真实 Val/Test 各两张图像产生 Study Mean 十类概率文件并核对 SHA-256。任何部分缺失时对应任务为 `BLOCKED`。

## CPU 验收与未完成项

- 全套 CPU/合成测试：**67 passed、0 failed、3 skipped**。新增 G/H 产物与调度门禁的针对性测试为 **6 passed**。
- 3 项 skipped：1 项真实 CUDA 图像推理；2 项 CUDA 中断恢复参数化测试。它们必须在朋友服务器的 `REMOTE_PREFLIGHT` 完成，不计入通过数。
- G/H 的真实 RTX 4090 前反向、恢复和 Val/Test 推理尚未在目标服务器执行，状态为 **PENDING**。本地 CPU 测试不能代替此项证据。
- 官方最终推理限制为全部权重不超过 500,000,000 字节、单图端到端不超过 100ms。4090 研究测速不能证明目标硬件合规；缺少官方口径实测时为 `LATENCY_UNVERIFIED`。

## 产物及安全

- 单任务结果包含最佳推理权重及 SHA、配置、每 Epoch 历史、Val/OOF（适用时）/无标签 Test Study 概率、逐类指标、manifest、状态和通过格式校验的 Submission 候选。G/H 全 Train 任务的 OOF 为 `N/A`。
- 所有结果直接保存在 `phase17_batch/` 分类目录；程序不创建 ZIP/TAR/GZIP，也不自动上传排行榜。
- 提交前对所有暂存文件运行代码专用安全检查，禁止原始胸片、私有权重、凭证和超过 2MB 的文件进入普通 Git 历史。
- 本次只修改独立朋友版 worktree；未启动正式训练，未修改本地 Phase 17 冻结实验。

## 发布结论

代码已达到**可交付朋友进行 REMOTE_PREFLIGHT**的本地标准。正式批量训练仍须由朋友在授权 8×RTX 4090 服务器完成预检并填写预算。GitHub push 尚需核对私有仓库 remote 地址并取得用户明确批准；本次不 push。
