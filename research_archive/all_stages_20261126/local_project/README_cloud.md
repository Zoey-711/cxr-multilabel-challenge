# Phase 17 云端运行手册（AutoDL）

本手册对应 2026-10-19 更换后的 AutoDL 实例 `autodl-container-97da5x9ggp-d597f2b8`。云端预检和测速不启动正式训练；D17、C17、S17 Fold 1 正式训练需用户批准。代码、原始 ZIP、权重、环境和新输出均计划放在数据盘 `/root/autodl-tmp`，避免占满约 30GB 的系统盘。

## 已核实的信息

- SSH：`ssh -p 39593 root@connect.westd.seetacloud.com`。密码只在本地交互终端输入，不存入项目文件。实例重建后以 AutoDL 页面新命令为准。
- GPU：服务器报告 NVIDIA GeForce RTX 4090，49,140 MiB，驱动 580.105.08；AutoDL 页面称 vGPU-48GB。数据盘 `/root/autodl-tmp` 在最近两次实测中仍为 **50GiB**，等待重启后复核扩容是否生效。
- 系统 Python：`/root/miniconda3/bin/python`，版本 3.12.3。新实例的 PyTorch、torchvision 与 CUDA 运行时版本尚未核实；项目应在数据盘中的隔离 `.venv` 安装比赛要求的 PyTorch 2.10.0、torchvision 0.25.0、CUDA 12.8，不改动驱动。
- 原始 ZIP：`F:\数据集\cxr_first.zip`，55,232,226,819 字节，SHA-256 `ff458640e9974d63aa3329ee7ae1248260b619223980218e3741aee4c6d6836f`。代码直接读 ZIP，不必解压。
- 云端项目 `/root/autodl-tmp/cxr-cloud`；数据 `/root/autodl-tmp/cxr_first.zip`；权重 `/root/autodl-tmp/cxr-pretrained/hub/checkpoints`；新输出 `/root/autodl-tmp/cxr-outputs`。

**容量闸门：**原始 ZIP 为 55.23GB，超过当前 50GiB 数据盘。只有重启后实测数据盘至少 80GiB，才开始上传。传完必须校验 SHA-256。云端包仅携带必需代码、配置和固定患者五折索引；无需复制冻结的本地历史实验。

## SSH 与文件核查

VS Code 安装 Remote - SSH 后使用上述 SSH 命令连接，部署完成后打开 `/root/autodl-tmp/cxr-cloud`。已用 2MiB 文件实测 Windows SFTP 的 `put -a` 可以续传。大文件上传前先确认数据盘容量；本地 PowerShell 使用 ASCII 临时盘符避免 SFTP 对中文路径的编码问题：

```powershell
subst Y: 'F:\数据集'
sftp -P 39593 root@connect.westd.seetacloud.com
```

在 SFTP 提示符内输入以下两行（不要输入提示符本身）：

```text
put -a Y:/cxr_first.zip /root/autodl-tmp/cxr_first.zip
quit
```

服务器核查：

```bash
df -h /root/autodl-tmp
sha256sum /root/autodl-tmp/cxr_first.zip
find /root/autodl-tmp/cxr-pretrained/hub/checkpoints -maxdepth 1 -type f -name '*.pth' -print
```

数据哈希必须与上文一致。云端预检会逐个核对冻结配置中的 ImageNet 权重哈希。密码、私钥、比赛 ZIP、模型权重和实验缓存不应上传 GitHub；如同步代码只使用私有仓库。当前部署直接使用 SCP。

## 安装、预检和短时测速

以下命令在云端终端运行。安装成功后不需要重复执行安装部分：

```bash
cd /root/autodl-tmp/cxr-cloud
export CXR_DATA_ROOT=/root/autodl-tmp/cxr_first.zip
export CXR_OUTPUT_ROOT=/root/autodl-tmp/cxr-outputs
export CXR_PRETRAINED_ROOT=/root/autodl-tmp/cxr-pretrained
export CXR_PYTHON_BIN=/root/miniconda3/bin/python
export CXR_INSTALL_FRAMEWORK=1
export TMPDIR=/root/autodl-tmp/cxr-tmp
mkdir -p "$TMPDIR" "$CXR_OUTPUT_ROOT"
bash scripts/cloud_install.sh --config configs/f17_densenet121_fold1_384.yaml
source .venv/bin/activate
python -m pytest -q tests/test_cloud_migration.py
```

安装脚本只创建项目 `.venv`。预检核对 Python、PyTorch、CUDA、GPU、原始 ZIP 哈希与 Train/Val 数量、患者五折索引、预训练权重和真实图像 FP16 前后向；失败即停止：

```bash
cd /root/autodl-tmp/cxr-cloud
source .venv/bin/activate
export CXR_DATA_ROOT=/root/autodl-tmp/cxr_first.zip
export CXR_OUTPUT_ROOT=/root/autodl-tmp/cxr-outputs
export CXR_PRETRAINED_ROOT=/root/autodl-tmp/cxr-pretrained
python scripts/cloud_preflight.py --config configs/f17_densenet121_fold1_384.yaml --gpu 0 --data-sha256 ff458640e9974d63aa3329ee7ae1248260b619223980218e3741aee4c6d6836f --output /root/autodl-tmp/cxr-outputs/d17_f1_preflight.json
python scripts/cloud_preflight.py --config configs/c17_fold1_384.yaml --gpu 0 --data-sha256 ff458640e9974d63aa3329ee7ae1248260b619223980218e3741aee4c6d6836f --output /root/autodl-tmp/cxr-outputs/c17_f1_preflight.json
python scripts/cloud_preflight.py --config configs/s17_fold1_384.yaml --gpu 0 --data-sha256 ff458640e9974d63aa3329ee7ae1248260b619223980218e3741aee4c6d6836f --output /root/autodl-tmp/cxr-outputs/s17_f1_preflight.json
```

上述每次预检会读取整个 ZIP，因此可能花几分钟。短时测速仅做数据读取与少量 GPU 优化步骤，不保存模型：

```bash
python scripts/cloud_benchmark.py --config configs/f17_densenet121_fold1_384.yaml --gpu 0 --output /root/autodl-tmp/cxr-outputs/d17_f1_benchmark.json
python scripts/cloud_benchmark.py --config configs/c17_fold1_384.yaml --gpu 0 --output /root/autodl-tmp/cxr-outputs/c17_f1_benchmark.json
python scripts/cloud_benchmark.py --config configs/s17_fold1_384.yaml --gpu 0 --output /root/autodl-tmp/cxr-outputs/s17_f1_benchmark.json
```

## 批准后只启动指定骨干和 Fold

先确认对应预检报告 `state=passed`。若服务器有 `tmux`，用 `tmux new -s phase17` 进入会话；断开按 `Ctrl+B`、`D`，恢复用 `tmux attach -t phase17`。本实例初查未安装 tmux，安装命令为 `apt-get update && apt-get install -y tmux`。若无法安装，可使用 `nohup` 或保持 SSH 连接。

```bash
cd /root/autodl-tmp/cxr-cloud
source .venv/bin/activate
export CXR_DATA_ROOT=/root/autodl-tmp/cxr_first.zip
export CXR_OUTPUT_ROOT=/root/autodl-tmp/cxr-outputs
export CXR_PRETRAINED_ROOT=/root/autodl-tmp/cxr-pretrained
python scripts/cloud_run_fold.py --config configs/f17_densenet121_fold1_384.yaml --model D17 --fold 1 --gpu 0 --preflight /root/autodl-tmp/cxr-outputs/d17_f1_preflight.json
```

C17 和 S17 分别使用 `configs/c17_fold1_384.yaml --model C17 --preflight /root/autodl-tmp/cxr-outputs/c17_f1_preflight.json` 与 `configs/s17_fold1_384.yaml --model S17 --preflight /root/autodl-tmp/cxr-outputs/s17_f1_preflight.json`，`--fold 1 --gpu 0` 保持。一次只启动一个骨干/折，不自动运行 15 个模型。异常退出后对同一命令增加 `--resume`；恢复模型、优化器、调度器、AMP scaler、随机数状态、Epoch、最佳分数和早停计数。

## 监控、备份和停止

```bash
nvidia-smi
find /root/autodl-tmp/cxr-outputs/experiments -maxdepth 2 -type f -name 'best_metrics.json' -print
tail -f /root/autodl-tmp/cxr-outputs/experiments/d17_f1_cloud/train_stdout.log
```

训练时用 `Ctrl+C` 请求安全停止；确认进程退出并备份 `last_training.pt`、`best_model.pt`、日志、验证/OOF 概率和 SHA 清单后再关闭实例。在本地 PowerShell 下载示例：

```powershell
New-Item -ItemType Directory -Force 'E:\计挑赛\cxr-multilabel-challenge\cloud_results' | Out-Null
scp -P 39593 -r root@connect.westd.seetacloud.com:/root/autodl-tmp/cxr-outputs/experiments/d17_f1_cloud 'E:\计挑赛\cxr-multilabel-challenge\cloud_results'
```

下载后比对云端清单与本地 SHA-256，再由用户在 AutoDL 控制台关机。若 SSH 地址或端口变化，以控制台当前值替换。
