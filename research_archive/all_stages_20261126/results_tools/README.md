# Phase17 朋友训练结果的 Windows 本地查看目录

本目录独立于 `E:\计挑赛\cxr-multilabel-challenge`，不会修改本地 Fold 训练、checkpoint 或比赛 Submission。结果仓库地址是 `https://github.com/Zoey-711/cxr-multilabel-results`；由于当前机器的 GitHub HTTPS 连接不可用，同步使用同一仓库的 SSH 地址 `git@github.com:Zoey-711/cxr-multilabel-results.git`。它只读取远端，不执行 Git push。

## 前提

请先在 GitHub 建立或确认该仓库是 **Private**，并确保当前 SSH 账号 `Zoey-711` 有读取权限。当前核验返回 `Repository not found`，所以**尚不能声称结果仓库已存在或已有结果**。朋友需要获授权后自行将经过脱敏的结果推送到此仓库；不能将原始胸片、私有路径、凭据或普通 Git 大文件权重加入结果仓库。训练状态快照可来自独立的只读 `batch_progress.py` 导出工具；朋友的训练进程无需停止。

本机需要 Git、Git LFS 和 PowerShell 5.1。同步脚本只下载 `CSV`、`JSON`、`JSONL`、`Markdown` 和轻量 `.log`；Git LFS smudge 默认关闭，随后只拉取这些扩展名。`.pth`、`.pt`、图像和压缩包不会默认下载。每个远端 commit 保存到独立的 `versions/<commit>/`，新版本验证通过才更新 `latest.json`。同步失败不会覆盖旧版本。若远端尚无文件，显示 `EMPTY`；仓库不存在、无权或网络失败显示 `NOT_AVAILABLE`。

Phase24 补充：同时按需下载 `.yaml`、`.yml` 和 `.sha256`，以保留实际训练配置及权重哈希记录；下载哈希文件不会下载对应权重。相同 commit 再次同步只复核已有版本，不创建重复版本。当前查看器自动展示上传的指标并核查概率格式与哈希；它不把上传的汇总数值当作本地独立复算结果。正式候选仍需原项目的 `import_phase17_friend_results.py` 验收及 `analyze_phase17_friend_import.py` CPU复算；缺少完整来源材料时保持未验收状态，不能绕过门禁。

## 每次查看只运行一条命令

在 Windows PowerShell 中执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\计挑赛\cxr-friend-results\Sync-FriendResults.ps1"
```

脚本会打印 26 项 `DONE/RUNNING/PENDING/FAILED/BLOCKED` 数量、已上传模型的 Val/OOF AUC 与最佳 Epoch、每份已同步概率 CSV 的 Study ID、10 类列顺序、概率范围和 SHA-256 校验结果。五折未齐时只显示 `PARTIAL_OOF`。`RUNNING` 阶段的指标是临时结果，不当作最终 AUC；Test 没有标签，也不计算 Test AUC。

可在 VS Code 中打开 [Phase17-results.code-workspace](Phase17-results.code-workspace)，左侧会同时显示独立结果目录与主项目，但同步脚本只写独立结果目录。工作区内还提供 **Tasks: Run Task → Sync Phase17 friend results**；该任务与上面的一条 PowerShell 命令等价。最新版本可从 `latest.json` 的 `path` 找到，Codex 可直接读取 `versions/<commit>/` 内已上传的轻量结果。

只对已经通过双方授权、确需本地推理的模型另行使用 Git LFS 按模型 ID 下载权重；不要对整个仓库执行无限制 `git lfs pull`。默认查看命令不会触碰模型权重。完整逐 Study 概率及模型权重如不适合放入结果仓库，应使用双方授权的独立传输方式。

## 校验范围与限制

- 每份 `*predictions.csv` 或 `*probabilities.csv` 要有 `Subject_id,Study_id,Prob_1,...,Prob_10` 的固定列顺序、非空且唯一的 Study ID、有限且处于 `[0,1]` 的概率。
- Val 预期 859 Study、Test 预期 1414 Study、完整五折 OOF 预期 18513 Study。单折 OOF 只按实际覆盖数报告；完整 OOF 还要求对应骨干五折均为 DONE。
- 优先比对同名 `.provenance.json` 中的 `csv_sha256`，其次比对同目录 `run_manifest.json`。缺少来源 SHA 会标记 `HASH_UNVERIFIED`，不虚报校验通过。
- 同步过程不访问朋友服务器、不暂停 GPU 任务、不上传任何文件。同步脚本无法判断尚未上传的真实训练状态；没有同步记录时显示 `EMPTY` 或 `NOT_AVAILABLE`。
- 下载版本仅供中途查看。后续若要把模型作为正式候选，仍应按原项目的导入工具核对训练 commit、患者划分、完整 manifest 和 checkpoint SHA-256。

## 本地合成验收

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\计挑赛\cxr-friend-results\tests\Test-ResultViewer.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\计挑赛\cxr-friend-results\tests\Test-SyncLocal.ps1"
```

第二个测试使用自己创建的临时本地 Git LFS 仓库，绝不连接 GitHub 或启动 GPU。它验证 LFS 中的 CSV 按需下载、权重不下载，以及远端读取失败时保留旧版本。
