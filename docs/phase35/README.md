# Phase35：两个已选方案补齐五折（8次新训练）

本批只新增Fold2–5，不再做Pilot筛选或调整超参数。原Phase32结果commit为`9fd6afc68b904ac4cf7f319053f09023507ca5ee`，历史N17基准结果commit为`cbfc5997438f371086cd1fbe232890590cd67ffb`。新代码应在独立克隆目录运行，不要在旧队列仍运行的目录拉取更新。

| 方案 | 复用的Fold1 | 新任务 | 唯一沿用的干预 |
|---|---|---|---|
| POS | P32_NS_POS | P35_POS_F02、F03、F04、F05 | EfficientNetV2-S，384；正类BCE权重=sqrt(阴性Study/阳性Study)，截断[1,5] |
| RES512 | P32_NS_512 | P35_RES512_F02、F03、F04、F05 | EfficientNetV2-S，512；原始BCE |

两组均使用合法ImageNet初始化、seed42、microbatch8×累积2=有效batch16、AdamW backbone LR=1e-4/head LR=3e-4、WD=1e-4、30轮上限、2轮warmup+cosine、Patience6、保守增强、Study Mean。EMA=.999，每次成功optimizer更新后更新，`use_buffers=True`；普通/EMA分别保存最佳权重。每折类别权重仅由该折4个训练Fold的**唯一Study**支持数计算，不复用Fold1的数值，也不读OOF/Val支持数拟合权重。初始化不使用Fold1微调权重。

训练入口保留朋友实际运行版`01fa68bdac65ce56d04a1fc99e8f5996ab09aa20`的Phase32训练代码原字节（包括已验证的PID侧文件启动修复），放在独立Phase35模块；Phase32/E08-Tune封存代码和配置保持不变。

## 一次准备

使用现有兼容环境，依赖沿用仓库`requirements-phase32.txt`（torch2.10.0、torchvision0.25.0、numpy、Pillow、PyYAML）。不安装驱动。新克隆后在该根目录激活环境；如果环境曾editable安装旧克隆，可执行`python -m pip install -e . --no-deps`切换到本克隆。

```bash
git clone --branch friend-gpu-batch --single-branch git@github.com:Zoey-711/cxr-multilabel-challenge.git cxr-phase35
cd cxr-phase35
# checkout交付消息给出的完整commit，固定本次代码版本
cp configs/phase35_server.example.yaml configs/phase35_server.local.yaml
```

填写`configs/phase35_server.local.yaml`：

- `data_root`：赛事原始ZIP，或保持官方布局的train.zip/valid.zip/test.zip目录。无标签Test是完整交付所需数据。
- `pretrained_root`：下层`hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth`，全SHA在每个任务配置中固定；缺少时不自动下载。
- `protocol_root`：原冻结patient_folds.csv、study_folds.csv、image_folds.csv，启动核对完整SHA，不重新划分。
- `baseline_results_root`：原phase17_batch目录，需要N17各折原始/EMA的配置、训练历史、Val/OOF分析文件；只读核验。
- `phase32_results_root`：原phase32_batch完整输出目录。必须保留P32_NS_POS和P32_NS_512的原始及EMA **best_model.pth**、配置、Val/OOF及manifest等冻结文件。仅下载Git小文件不足以满足复用条件；若原目录不在服务器，先按已授权的Phase32私有Release传输补齐，程序不自动重训。
- `output_root`：独立路径，末级为`phase35_batch`，不可放进旧结果目录。
- `gpu_ids`、`max_concurrent`：明确授权的物理卡号，最多8；每卡最多一个独立任务。与Phase32/E08-Tune共享主机锁，不能两个队列同时各占满8卡。
- `gpu_hours`：必须填写朋友授权的总GPU小时预算；空值拒绝启动。包括预检、Fold1缺失Test推理和正式任务。
- `upload.enabled`：希望自动回传时由朋友设为`true`，并填写独立`upload.checkout`。需要该服务器已经完成`gh auth login`、Git提交身份和私有仓库写权限；不传递任何令牌到配置。

8并发预检要求至少32逻辑CPU、64GiB可用内存、40GiB输出空间，数据和原历史模型另计；未达到时减少授权并发，不自动更改训练batch。现有同机器Fold1训练阶段分别耗时约2.357/2.380 GPU小时；简单外推8次约18.95 GPU小时，尚未含新折早停变化、预检、全部推理与分析，不能视为承诺。每个任务保守预留12小时，全部8卡同时启动需可用预算至少96小时加预检开销；这是资源保护上限，不是预计实际消耗。较低预算会降低实际同时启动的任务数，余额不足的任务保留PENDING，不会越权启动。已完成任务不会重复计费；异常中断无法确定消耗时保守计入一次预留额。

## 唯一推荐启动命令

```bash
python scripts/phase35_run.py --config configs/phase35_server.local.yaml
```

先进行强制REMOTE_PREFLIGHT：核对代码/配置/数据/预训练/冻结Fold1与历史基准；逐卡轮流执行真实短步训练、AMP、checkpoint恢复、Val/Test短推理及显存检查。再将两个Fold1的纯推理权重和现有概率复制到新输出目录，补做缺失的普通/EMA Test推理（不训练Fold1）。全部通过后自动执行8项训练。失败的预检保留记录，有限重试；不能绕过门禁。

每个任务结束立即保存完整本地产物。Test推理异常时保留已完成训练和Val/OOF，恢复时仅继续缺失阶段。新任务初始化依然来自ImageNet，不能加载Fold1作为新折起点。每轮保存完整模型、optimizer、scheduler、GradScaler、随机数、EMA、早停和最佳记录；中断后从最后完整Epoch恢复。OOM/非有限值不自动调参。完成结果不重训。

可选操作（同一配置）：

```bash
python scripts/phase35_run.py --config configs/phase35_server.local.yaml --action preflight
python scripts/phase35_run.py --config configs/phase35_server.local.yaml --action status
python scripts/phase35_run.py --config configs/phase35_server.local.yaml --action stop
python scripts/phase35_run.py --config configs/phase35_server.local.yaml --action resume
```

`preflight`只短测及复用Fold1的Test推理，不启动8项正式训练；`stop`为排空停止：当前任务安全结束，不派发新任务。正常中断后重复推荐命令自动复用有效预检、DONE及恢复状态；显式STOP后使用`resume`。不可在仍有孤儿训练子进程时重复启动。确定性的文件损坏、预算不足等会明确报告，不能删除旧结果来强行重跑。

## 结果目录及统计协议

```text
phase35_batch/
  preflight/                    # 强制短测报告、用量账本
  reused_fold1/P32_NS_POS/      # 校验后复制原Fold1；不修改源目录
  reused_fold1/P32_NS_512/
  experiments/P35_POS_F02/     # 其余7项同级
  queue_logs/
  reports/experiment_status.csv
  reports/experiment_summary.csv
  reports/five_fold_summary.csv
  reports/five_fold_analysis.json
  reports/delivery_manifest.json
  oof/{POS,RES512}_{raw,ema}_all.csv
  validation/{POS,RES512}_{raw,ema}_fivefold.csv
  test_predictions/{POS,RES512}_{raw,ema}_fivefold.csv
  analysis/per_class_metrics.csv
  deployment/cost_comparison.csv
  deployment/resolution_comparison.json
```

每项包含普通/EMA最佳纯推理权重、配置、恢复状态、Val/OOF/Test概率、逐类AUC/AP和支持数、训练历史、manifest、SHA-256、DONE或失败记录。OOF按患者所属Fold拼接，全部覆盖18513 Study/11055患者才称完整；不平均不同Fold的OOF。不足五折标记INCOMPLETE。

普通/EMA分别对比同一种N17基准；主要关注EMA。完整OOF及排除Fold1的Fold2–5分别做2000次有效患者配对Bootstrap，seed42，单类无法计算AUC的抽样丢弃并计数。Val先等权平均五折Study概率再计算AUC，不平均五个AUC。Fold1用于挑选两个方向，完整OOF含选择偏差；后四折用于确认。官方Val已重复用于早停，不作为新独立测试。Test只生成概率，无Test AUC、无Submission。

部署报告统计每个真实推理文件及配置、五折总大小，记录384/512 Conv2d+Linear MACs及端到端时延。研究测速包括ZIP读取、解码、预处理、传输、前向和CSV写出，batch1，5张预热＋20张×3轮；这不是官方100图协议，服务器并发负载可能不同。所有候选保持`T4_UNVERIFIED`，不能用4090结果声称T4≤100ms。单方案五折EMA大约408MB，两个五折方案一起部署会超500MB；最终以实际文件计数为准，不生成新的融合搜索。

## 结果自动回传与本地同步

朋友在配置明确开启上传后，代码向同一私有仓库的**新分支**`results/phase35-fivefold`写入`results/phase35_batch`。纯推理权重逐文件上传到**新Release** `phase35-fivefold-v1`，资产名例如`P35_POS_F02__ema_best_model.pth`，SHA及大小保存在`binary_release_manifest.json`；不压缩、不传原图/原始标签文件/optimizer状态。遇到已存在不同SHA的Release资产拒绝覆盖，不使用force或clobber。

本地产物每完成一项立即持久化；结果上传在当前GPU任务批次排空后和最终分析后执行，以免网络阻塞队列的预算记账。上传失败记录`reports/UPLOAD_FAILED.json`，保留所有完成结果。之后可只重试上传而不训练：

```bash
python scripts/phase35_run.py --config configs/phase35_server.local.yaml --action upload
```

用户Windows端可将结果分支克隆到独立目录（Git中没有模型权重，默认不会下载全部权重）：

```powershell
git clone --single-branch --branch results/phase35-fivefold git@github.com:Zoey-711/cxr-multilabel-challenge.git "E:\计挑赛\cxr-friend-results\phase35-results"
git -C "E:\计挑赛\cxr-friend-results\phase35-results" pull --ff-only
git -C "E:\计挑赛\cxr-friend-results\phase35-results" rev-parse HEAD
```

首次只执行clone；以后执行pull和rev-parse。工作目录有自己的修改时先保留，不执行reset/clean。分析时记录commit及`export_manifest.json`，历史Git提交仍可追溯。模型需要时按`binary_release_manifest.json`单独通过已认证的`gh release download phase35-fivefold-v1 --repo Zoey-711/cxr-multilabel-challenge --pattern '指定模型__ema_best_model.pth' --dir '独立下载目录'`获取并校验SHA。

本地CPU验收证据见[acceptance.md](acceptance.md)。真实Linux、8卡分配/共享IO吞吐、512显存与CUDA恢复仍由强制远程预检确认，不声称本机已经验证4090。
