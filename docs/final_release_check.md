# Phase 17-Batch 22项发布前检查（2026-10-20，历史记录）

此报告对应提交 `779327f` 的 22 项版本；当前 26 项计划另见 `phase17_26_acceptance.md`。

结论：代码可进入私有 GitHub 仓库的待发布状态；朋友服务器正式训练仍被 `REMOTE_PREFLIGHT` 与正数 GPU 小时预算双门禁阻止。本报告不代表 RTX 4090 实机检查已通过。

## 本地隔离验证

- 在独立 `friend-gpu-batch` worktree 中设置 `CUDA_VISIBLE_DEVICES=-1`，运行 `pytest tests -q -ra -p no:cacheprovider`：**61 passed、0 failed、3 skipped**。
- 跳过项为 `test_archive_test_inference_keeps_study_ids_and_has_no_labels` 和 `test_full_epoch_resume_is_identical` 的两个 worker 参数。这 3 项必须在朋友服务器的 `REMOTE_PREFLIGHT` 中实际通过；跳过、失败或缺少 XML 记录都不签发通过凭证。
- 原项目 S17-F1 自然早停：第14轮停止，第8轮最佳 Study Macro-AUC 0.79479172828591。未修改其训练代码、配置、日志或权重。
- CPU 测试证据保存在本地忽略目录 `outputs/phase17_batch_release_20261020/regression_verified.log` 与 `regression_verified.xml`；这些文件不会上传 GitHub。
- Git 暂存文件审计：129 个跟踪文件、约 0.55 MB 代码和文档，无原始图像、比赛 ZIP、私有权重、访问凭证或单文件超过 2MB 的大型产物；`git diff --cached --check` 无空白错误。
- `batch_run.py --action validate` 读取冻结计划成功：22 项基础任务、4 项禁用扩展槽位。README 的远程预检、正式启动、状态、停止及 `clear-stop` 恢复命令均对应实际脚本参数。

## 训练和部署门禁

- 官方权重总大小 `≤500,000,000` 字节，目标环境完整单图延迟 `≤100ms`；80ms 为内部安全目标。边界 100.000ms 通过延迟数值门槛，100.001ms 失败。缺少官方同口径实测为 `LATENCY_UNVERIFIED`。
- `friend_remote_preflight.sh` 独立执行数据/权重/折划分校验、真实单卡训练和 Val/Test 短推理、3 项 CUDA 测试、按 1→2→4→授权并发数逐级设备检查。实际 RTX 4090 结果：**PENDING**。
- `friend_start.sh` 不再自动执行预检。`batch_run.py` 检查通过凭证的主机、计划哈希、环境路径、GPU、逐级并发记录、真实数据与 CUDA 测试。还必须提供正数的朋友授权训练 GPU 小时预算。
- 本次一键入口仅包含冻结的 22 项基础任务。G/H 四项保持 `PENDING`，不会由一键入口启动。
- 所有结果直接保存在分类文件夹；不自动打包、不上传排行榜。

## 远端待核验

1. 单机 8×RTX 4090 的实际显存、CPU、RAM、磁盘、ZIP 随机读取速度与并发资源是否满足授权配置。
2. Python/PyTorch/torchvision/CUDA 环境、合法 ImageNet 权重 SHA、固定折索引和真实 Train/Val/Test 读取。
3. 三项 CUDA 测试与逐级并发阶段；正式批量训练前由朋友确认 GPU 小时预算。
4. 官方目标硬件的完整推理计时口径与实测；4090 测速仅供研究，不证明比赛部署合规。

没有通过以上远端预检时，状态为 `NOT_READY_FOR_TRAINING`，不能把本地 CPU 测试当成远端训练许可。
