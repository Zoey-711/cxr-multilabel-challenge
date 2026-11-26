# 远程 26 项实验状态：UNKNOWN

截至 2026-10-21，本地未发现朋友服务器同步的正式 `phase17_batch/experiments`、任务状态或 `DONE.json`。独立 worktree `E:/计挑赛/cxr-friend-gpu-batch-expanded/outputs/` 中发现的是本地验收、预检和测试产物，不能作为朋友实际训练的证据。本轮没有连接朋友服务器，也没有修改该仓库或队列。

**判断：UNKNOWN**。既不能推断 `NOT_STARTED`，也不能推断 `TRAINING` 或 `COMPLETED`。来源：上述本地目录只读扫描；朋友服务器状态不可访问。

若尚未启动，朋友按已发布的 `README_friend.md` 准备合法比赛 ZIP、冻结协议文件、合法 ImageNet 权重、数据/输出路径、GPU 数量与三个明确授权的 GPU 小时预算，然后在自己的服务器运行：

```bash
bash scripts/friend_run_all.sh --config configs/friend_server.local.env
```

入口会先进行强制 `REMOTE_PREFLIGHT`；是否实际通过、是否启动任务，只能以服务器端预检回执和任务状态为准。本地无需且不得代朋友启动。来源：`E:/计挑赛/cxr-friend-gpu-batch-expanded/README_friend.md`。
