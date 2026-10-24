# 朋友服务器交付：Phase32 与 E08-Tune

请在**同一台Linux服务器、独立克隆目录**中使用本文件所附的 `friend-gpu-batch` 完整commit。比赛数据、冻结患者五折文件、合法ImageNet权重和历史Phase17结果经授权渠道单独放置；克隆代码不会得到这些文件。不要在仍运行旧Phase17队列的目录直接改代码。

先在独立目录克隆私有仓库并检出交付消息中的完整commit：

```bash
git clone --branch friend-gpu-batch git@github.com:Zoey-711/cxr-multilabel-challenge.git cxr-friend-delivery
cd cxr-friend-delivery
git checkout <交付消息中的完整commit>
cp configs/phase32_server.example.yaml configs/phase32_server.local.yaml
cp configs/e08_tune_server.example.yaml configs/e08_tune_server.local.yaml
```

填写两个忽略Git的本地配置：`configs/phase32_server.local.yaml` 和 `configs/e08_tune_server.local.yaml`。两个 `output_root` 必须是互不嵌套的独立目录，例如同一磁盘下并列的 `phase32_batch` 与 `e08_tuning`。分别填写 `data_root`、`pretrained_root`、`protocol_root`、`baseline_results_root`、`gpu_ids`、`max_concurrent` 和已批准的GPU小时预算；E08配置还须填写 `extension_gpu_hours`，填0表示不补折。**两个预算相加**应在朋友授权的总资源内。两个配置的 `upload.checkout` 也要用不同的独立目录。

默认**依次执行**。两条运行入口现在共用Linux `/tmp/cxr_friend_gpu_queue.lock`，锁从真实预检一直保持到训练、评价和结果导出结束；另一队列同时启动会直接报错并退出。每项训练仍由自己的 `CUDA_VISIBLE_DEVICES` 绑定一张GPU，每卡最多一个任务。这个锁覆盖本批两套入口；如另有旧Phase17或其他GPU任务，请等它们结束再执行。无需同时启动两套脚本，也没有跨队列并发调度器。两批产物、日志、内部queue.lock、GPU小时账本和结果上传分支各自独立。

先运行静态配置核对，再运行真正的服务器预检；`--action preflight`执行短时真实数据与CUDA步骤，但**不启动正式长训练**。它会计入配置的GPU小时预算，并在同一输出目录内留下可复用凭据。

```bash
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action validate
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action preflight
python scripts/phase32_run.py --config configs/phase32_server.local.yaml
```

Phase32八项结束、没有该队列GPU任务运行后，再执行E08-Tune：

```bash
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action validate
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action preflight
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml
```

E08-Tune默认先运行5项Fold1 Pilot；只有预定患者级OOF条件成立、补折预算仍足够时，最多两种方法各补Fold2–5。冻结N17基准直接复用，不重训。两套 `--action preflight` 均会核验数据/权重/患者划分，并进行真实短步前反向、AMP、checkpoint恢复和短推理。预检失败不启动正式训练。4090 CUDA结果必须在朋友服务器实测；本地CPU验收不代替它。

查看状态或排空停止：

```bash
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action status
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action stop
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action status
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action stop
```

异常中断后，对相应入口加 `--action resume`；它复用已通过的预检、已完成模型和Epoch恢复状态。停止命令只阻止领取新任务，不强杀当前训练。不要手工清理输出目录或锁文件；`/tmp` 主机锁在进程退出时由系统释放，锁文件本身保留以避免竞态。

结果默认只保存在朋友服务器。若朋友明确授权增量上传，在各自本地配置将 `upload.enabled` 设为 `true`，设置可写的同一**私有**GitHub仓库、独立 `upload.checkout` 路径；朋友服务器需先完成 `gh auth login`。Phase32只允许 `results/phase32-pilots` 分支，E08只允许 `results/e08-tune` 分支。每个任务DONE后上传小型配置、指标、Val/OOF概率及哈希；模型权重、原始影像、原始标签、完整Test概率不进入普通Git结果分支。上传失败写 `UPLOAD_FAILED.json`，本地训练结果保留；大权重另按双方授权的方式交付。

运行详情：[Phase32](phase32/README.md)、[E08-Tune](e08_tune/README.md)。本文件没有声称朋友端预检或正式训练已经完成，也没有T4延迟合规结论。

# 补充说明：本次上传的实验与结果位置

本说明对应 `friend-gpu-batch` 分支中已上传的 Phase32 与 E08-Tune 代码；已核对的代码提交为 `30a499426d9758810fcb464025e73d80e16b1159`。朋友应使用该分支最新的指定提交，在**新的克隆目录**运行，不要更新仍在执行旧任务的工作目录。本地 CPU/合成测试通过不代表朋友服务器上的 4090 预检或正式训练已完成。

| 队列 | 本次正式训练 | 基准与扩展 | 要解决的问题 |
| --- | --- | --- | --- |
| Phase32 | 8 个 Fold1 Pilot | 复用历史 CS17、NM17、N17 的同折结果；本批不自动补折 | 大骨干学习率、N17 输入分辨率、ASL、类别加权、亮度增强与 Mixup 的单变量比较 |
| E08-Tune | 5 个 Fold1 Pilot | 直接复用 N17_F01 基准；仅当预定 OOF 条件和单独授权预算均满足时，最多再训练 8 个 Fold2–5 模型 | N17 EMA decay、学习率及延长训练日程是否改善未见患者的表现 |

这两套队列**依次运行**，建议先 Phase32、后 E08-Tune。它们使用同一台服务器的 GPU 总锁，不能分别启动后同时占满 8 卡。两份本地配置的 `output_root` 必须互不嵌套，末级分别为 `phase32_batch` 和 `e08_tuning`；`gpu_hours` 分别填写朋友明确同意的上限，E08 还须填写 `extension_gpu_hours`。配置文件中的示例路径不是可用数据路径；比赛数据、合法 ImageNet 权重、冻结五折索引和历史对照结果需经授权单独放置。GPU 数量与并发上限以朋友实际空闲资源为准。

在仓库根目录，安装现有兼容环境并填写两份配置后，分别执行以下**一条正式入口**。每条入口自身先完成 REMOTE_PREFLIGHT，全部通过后才启动该队列；也可以先单独运行 `--action preflight` 检查。不要跳过门禁。

```bash
python scripts/phase32_run.py --config configs/phase32_server.local.yaml
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml
```

每个完成任务的权重、配置、逐轮日志、Val/该折 OOF 概率、逐类指标和 SHA-256 保存在各自 `output_root` 下的 `experiments/<实验ID>/`；E08-Tune 按 `pilot/experiments/` 与 `expansion/experiments/` 分开。Phase32 的队列状态在根目录 `queue_state.json`，E08-Tune 的两个阶段分别在 `pilot/queue_state.json`、`expansion/queue_state.json`；阶段总表各自在 `reports/experiment_summary.csv` 和 `reports/experiment_status.csv`。`DONE.json` 表示该任务的评价及导出均成功，`FAILED.json` 表示失败；未完成的 Fold1 OOF 只能称为部分 OOF。E08-Tune 仅在某变体五折真正齐全时才会生成顶层完整 OOF、五折 Val 概率及 `reports/five_fold_summary.csv`。两套实验均**不**自动生成比赛 Submission，也不计算隐藏 Test AUC。

可以随时只读查看，或在当前任务完成后排空停止；中断后用同一输出目录恢复，不要删除 checkpoint 或锁文件：

```bash
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action status
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action stop
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action resume
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action status
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action stop
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action resume
```

**结果上传默认关闭。** 如朋友已明确授权，可在两份配置中分别启用 `upload.enabled` 并指定私有结果分支和独立 checkout；完成任务后增量上传的仅是小型指标、配置、Val/OOF 概率及哈希。模型权重、原始图像和完整 Test 概率仍留在朋友服务器，需另走双方授权的传输方式。上传失败会留下 `reports/UPLOAD_FAILED.json`，不会删除本地结果。4090 延迟是研究数据；没有目标 T4 的完整端到端实测，部署状态仍为 `LATENCY_UNVERIFIED`。
