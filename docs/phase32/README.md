# Phase32：8个受控Pilot（朋友Linux RTX4090服务器）

本机只进行了CPU与合成数据检查，没有启动正式训练。代码开发使用独立副本，分支为 `friend-gpu-batch`。历史结果来源固定为 `cbfc5997438f371086cd1fbe232890590cd67ffb`；`history_inventory.json`保存26项真实配置与训练历史的哈希和指标。不要在仍运行旧任务的目录执行 pull；为本批新建克隆目录。

## 实验与历史对照

| ID | 历史同Fold基准 | 唯一研究变量 | 假设 |
|---|---|---|---|
| P32_CS_LR | CS17_F01，ConvNeXt-Small | backbone LR 1e-4→2.5e-5，head LR仍3e-4 | 更保守地微调大骨干，减少早期遗忘 |
| P32_NM_LR | NM17_F01，EfficientNetV2-M | 同上 | 检验大骨干欠佳是否与微调步长相关 |
| P32_NS_448 | N17_F01，EfficientNetV2-S+EMA | 384→448 | 检验空间细节收益 |
| P32_NS_512 | N17_F01，EfficientNetV2-S+EMA | 384→512 | 同上，明确额外部署成本 |
| P32_NS_ASL | N17_F01 | BCE→ASL，γ−4/γ+1/clip .05/eps 1e-8/focal detach | 检验该骨干对负类梯度抑制的响应 |
| P32_NS_POS | N17_F01 | BCE正类权重：sqrt(阴性Study/阳性Study)，截断[1,5] | 检验稀有类别的受控梯度重加权 |
| P32_NS_PHOTO | N17_F01 | brightness/contrast应用概率 .5→0 | 检验保留原始亮度是否更稳健；几何增强不变 |
| P32_NS_MIXUP | N17_F01 | microbatch内Mixup，Beta(.2,.2)，概率1 | 检验软监督正则化；不跨训练/OOF患者混合 |

所有实验固定Fold1、seed42、ImageNet初始化、10类、Study Mean、AdamW、weight decay=1e-4、cosine+2轮warmup、最多30轮、Patience=6、strict improvement/min_delta=0。microbatch=8、accumulation=2、有效batch16。N分支保留原EMA=.999，每次成功optimizer更新，use_buffers=True；普通和EMA分别保存及复算。类别权重仅由四个训练折的唯一Study计数生成，不读取Val/OOF标签支持数。

合法ImageNet来源沿用旧CS17/NM17/N17的torchvision权重；完整SHA已写入各配置，启动时逐个检查，缺失不自动下载。朋友须按赛事授权渠道获得数据与这些权重，不使用任何额外医学数据。

旧26项已包含这三个架构的384 BCE训练及全Train/蒸馏，但没有本表的8个精确干预。不能称ASL和512分辨率是项目首次尝试：本地L01是ConvNeXt-Tiny ASL（Val 0.79853963，best5），R01是ConvNeXt-Tiny 512（Val 0.79565350，best4）；两者对原基准的配对区间均跨0。本批仅测试它们迁移到EfficientNetV2-S是否有效，不据此宣称已有稳定收益。已扫描本地configs与26项resolved_config，未发现同骨干448、上述正类加权、Mixup或移除光度增强的同条件记录；不等于所有未同步实验均不存在。

旧记录来源：主项目 `outputs/analysis/phase10_l01_pipeline_20261017/independent_validation_stdout.log`、`phase11_r01_pipeline_20261017/independent_validation_stdout.log` 与各自 `fixed_comparison_stdout.log`。这些原始日志不上传代码仓库。

## 数据和环境准备（一次）

克隆私有仓库friend-gpu-batch，并checkout交付消息中的完整commit。环境使用Python3.12/3.13及现有兼容CUDA的torch2.10.0、torchvision0.25.0。不要替换驱动。可在独立venv中运行 `python -m pip install -e . --no-deps`；其余依赖见 `requirements-phase32.txt`，仅在缺失时安装。真实Linux/CUDA/4090检查尚待服务器执行。

复制 `configs/phase32_server.example.yaml` 为 `configs/phase32_server.local.yaml`，填入：

- `data_root`：官方完整ZIP，或包含train.zip/valid.zip/test.zip的目录，内部保持官方cxr_first布局。任意二进制切片不等于独立ZIP；当前入口不支持直接读取尚未合并的 `.part` 文件。
- `protocol_root`：包含冻结patient_folds.csv、study_folds.csv、image_folds.csv的目录。程序只复制哈希匹配的原划分到新克隆的ignored outputs目录，不重新划分。
- `pretrained_root`：下层有hub/checkpoints及对应三个ImageNet文件。
- `baseline_results_root`：历史phase17_batch目录，保留CS17_F01/NM17_F01/N17_F01的resolved_config、train_history、原始validation_analysis/oof_analysis（含标签的合法评估文件），N17还需EMA对应目录；只读引用。
- `output_root`：全新独立路径，末级必须为phase32_batch。
- `gpu_ids`、`max_concurrent`、`gpu_hours`：朋友明确授权。默认预算null必定阻止启动；不得把示例理解为已授权。

8并发至少需要32个可用逻辑CPU、64GiB可用内存、100GiB空闲输出空间（另加数据和预训练权重）。每卡最大预检reserved显存22GiB，实际显存和吞吐PENDING。预算按GPU小时计，不是墙钟小时。每个正式任务预留12h，8个最多96h，加最多2h短测；这只是保护上限，并非预计耗时。历史同机器记录约1–3.3h/任务，但新分辨率、增强、早停和共享IO可能改变耗时；第一次实机后再估计，不承诺8卡线性加速。

## 唯一推荐启动命令

在新克隆的根目录、已激活环境中运行：

```bash
python scripts/phase32_run.py --config configs/phase32_server.local.yaml
```

自动核验配置/源代码SHA、授权预算、CPU/RAM/磁盘、空闲RTX4090、合法权重、冻结患者隔离和历史基准。然后顺序执行8项真实单步预检，每项覆盖前后向、AMP、优化器/调度器/scaler/RNG恢复、2张Val及2张无标签Test短推理；各授权卡轮流绑定。全部通过后动态调度最多8个独立GPU任务，不使用DDP。不存在允许跳过预检的参数。OOM/非有限梯度不自动修改配置。较低预算可以只启动预算允许的任务，剩余保持PENDING。

状态、排空停止、恢复：

```bash
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action status
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action stop
python scripts/phase32_run.py --config configs/phase32_server.local.yaml --action resume
```

stop不杀死正在训练的模型，当前任务完成后停止领取。异常后同命令恢复最近完整Epoch；不会重训DONE任务。死锁只在同主机父子进程均已退出时清理。中断期间预算保守按预留上限计入一次；不会漏算或因重复恢复反复扣减。预检失败不进入训练；保留全部失败证据，修复环境后用新的输出目录重新预检（不得把旧产物删除后假装未花费预算）。

## 输出与结果接收

每项独立目录 `phase32_batch/experiments/P32_*/`：best_model.pth、best_model.sha256、原last_training.pt完整恢复状态、resolved_config.yaml、train_history.csv（每轮Study Val AUC）、train.log、metrics.json、per_class_metrics.csv、val_predictions.csv、oof_predictions.csv、run_manifest.json、environment.json、model_size.json、latency.json、DONE.json或FAILED.json；EMA独立前缀文件。训练中每轮metrics.jsonl和status.json即时可读，最佳权重即时保存。

完成一项立即生成标准Study概率、独立指标、哈希、配对历史基准Bootstrap；无须等所有任务完成。OOF始终只属于Fold1：3687 Study/2211患者，绝不冒充完整五折；Val859 Study/228患者。不计算Test AUC、不生成比赛Submission。Test仅用于无标签预检的2图输出。

`reports/experiment_summary.csv/json`、`experiment_status.csv`、`phase32_report.md`、`delivery_manifest.json`随完成任务更新。普通/EMA分开列示。bootstrap为患者级2000次有效抽样，seed42，无效单类抽样跳过并计数。继续标准：相对对应Fold1旧基准OOF≥.002、95%差值区间下界>0、最差类下降不超过.02；仅形成探索性推荐，绝不自动增加训练。官方Val反复选模、多项Pilot比较及Fold1选择偏差必须保留。

推理权重以真实文件字节计，配置另计。≤500000000字节是大小门槛；4090研究测速不覆盖官方写出范围、不代替T4端到端≤100ms，状态为LATENCY_UNVERIFIED或SIZE_FAIL。不能凭AUC将大模型认定为可部署。

不压缩、不自动删除结果。默认不上传。需要增量上传时，朋友在同一配置中明确启用 `upload.enabled: true`，使用已登录的gh、具有写权限的同一私有仓库和独立 `results/phase32-pilots` 分支，`checkout`必须指向独立新目录。只传完成任务的小型指标、Val/OOF概率、配置和哈希；不传图像、原始标签、权重、Test预测或凭据。不修改代码分支或历史results/phase17-20261024。冲突或上传失败保留本地结果，在reports/UPLOAD_FAILED.json报告，不影响其他训练。

启用上传后，本地从该私有仓库的 `results/phase32-pilots` 分支下 `results/phase32_batch` 同步；没有该分支或真实文件就报告NOT_UPLOADED，不能当作任务已完成。大权重另经授权交付。
