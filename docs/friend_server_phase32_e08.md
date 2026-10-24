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
