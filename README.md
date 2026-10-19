# Phase17 Batch 私有仓库交付

分支 `friend-gpu-batch` 在独立 Git worktree 准备。朋友通过私有仓库克隆代码，在一台 8×RTX 4090 Linux 服务器上运行；不会通过 SSH 访问朋友设备。原本地训练项目不受影响。

- [朋友三步操作和启动前阻塞](README_friend.md)
- [22 次任务与最多 4 个关闭的条件槽位](docs/expanded_plan.md)
- [统一任务配置](configs/batch_plan.yaml)
- [结果文件夹、校验与离线复算](README_results.md)

基础扩展档共 22 次：D17、C17、S17、N17 各 5 折，加 ConvNeXt-Small 与 EfficientNetV2-M 的 Fold 1 Pilot。原 15 次训练配置及患者折 SHA-256 保持冻结。N17 采用 N01 固定 EMA 规则；C17 的 EMA 旁路延续原方案。单折 OOF 只与相同 Fold 的患者及 Study 比较。

两个 Pilot 的 torchvision 官方 ImageNet 权重已在独立 worktree 下载并计算完整 SHA-256，已写入配置和计划；预检仍会对朋友收到的文件逐字节核对。朋友已确认单机 8×RTX 4090；GPU 小时预算、CPU、内存、磁盘和官方目标硬件计时口径仍待确认。条件 4 次扩展处于关闭状态。

最终部署同时要求推理权重总大小≤500,000,000 字节、完整单图端到端延迟严格<100ms。缺乏官方硬件、精度、批量与计时口径时均标记 `LATENCY_UNVERIFIED`。`PASS` 才可视为部署候选；4090 或本机 4060 的研究测速不能替代目标环境验证。FP16、量化、蒸馏要对实际产物重新验证 AUC、大小和延迟。

目前只完成 CPU 合成流程验收；真实 4090 前反向、吞吐、延迟和 Linux 环境属于朋友获授权后的启动前预检。自动 Test 推理仅使用无标签比赛 Test，输出完整概率及四列提交候选，但不会上传平台。所有结果直接保存在文件夹，不生成压缩包。数据、私有折索引、ImageNet 权重及结果通过赛事允许的独立渠道传递，不进入 Git。
