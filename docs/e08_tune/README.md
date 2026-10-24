# E08组成模型专项调参：先5项Pilot，条件满足后最多8项补折

本批仅优化EfficientNetV2-S EMA训练。保持E08公式 `0.5 E03 + 0.5 EfficientNet EMA`；不搜索融合权重，不使用Test成绩，不自动生成或提交Submission。使用新克隆/新输出目录，不改动正在执行的Phase17/32代码。开发及CPU测试发生在Windows；正式任务只能在朋友授权的Linux RTX4090上运行。

## 真实配置核验

详细证据在 `source_audit.json`：来自结果commit `cbfc5997438f371086cd1fbe232890590cd67ffb` 的N17五折配置、历史、Val/OOF概率和manifest均记录SHA-256；同时核对本地E08 manifest所引用N01配置哈希。

| 项目 | N01-EMA | N17 EMA |
|---|---|---|
| 训练人群 | 全部18513 Study/11055患者 | 每折其余四折；Fold1训练14826 Study/8844患者 |
| OOF | 不适用 | 对应未见患者Fold；完整18513 Study |
| 输入/预训练 | 384，torchvision合法ImageNet | 相同 |
| AdamW学习率 | backbone 1e-4 / head 3e-4 | 相同 |
| weight decay | 1e-4 | 相同 |
| 损失/增强 | BCE，保守几何及亮度对比度增强 | 相同 |
| batch/accumulation | 8×2=16 | 相同 |
| 日程 | 30轮上限，cosine，warmup2轮 | 相同 |
| early stopping | raw Study Val AUC，max，min_delta0，patience6 | 相同 |
| EMA | .999，每次成功optimizer更新，use_buffers=True | 相同 |

EMA第一次更新复制模型，随后按声明decay更新；AMP跳步不更新EMA或scheduler。BatchNorm缓冲区沿用torch AveragedModel的use_buffers=True行为，不另做BN重新校准。普通与EMA最佳Epoch分别保存；停止仍由普通模型Val控制，**不是改为EMA早停**。

N01实际运行9轮，普通最佳Epoch3/Val .792584706；EMA最佳Epoch5/Val .804906183。E08 .811668646是融合Val，不是N01单模型成绩。N17 EMA Fold1最佳Epoch5/Val .799543845/OOF .804189658；五折EMA最佳Epoch依次5、4、4、4、5。复算完整OOF .807672123；五折**概率平均后的Val AUC .806836246**，不是五个Val AUC的算术平均 .800939389。

N01与N17参数相同，但人群、训练平台和代码来源不同，不能声称逐位复现。历史环境分别为4060/4090、torch2.10.0+cu128；N17元数据显存报告48508MiB，与“24GB卡”描述不同，须朋友实机核实。新入口始终检查实际GPU及至少22GiB容量，不把历史资源记录当实测保证。

## 固定实验

原始基准直接复用N17_F01及其他冻结折，不重复训练。最多六种比较，实际只有五个新Pilot：

| 变体 | 唯一因素 | 具体参数 | Fold1任务 |
|---|---|---|---|
| 基准（复用） | 无 | EMA .999，LR 1e-4/3e-4，30轮 | N17_F01，不入训练队列 |
| DECAY_LOW | EMA decay | .998 | ET_DECAY_LOW_F01 |
| DECAY_HIGH | EMA decay | .9995 | ET_DECAY_HIGH_F01 |
| LR_LOW | 两参数组同比LR缩放 | 7e-5 / 2.1e-4 | ET_LR_LOW_F01 |
| LR_HIGH | 两参数组同比LR缩放 | 1.3e-4 / 3.9e-4 | ET_LR_HIGH_F01 |
| LONG | 训练/余弦日程长度 | 上限40轮，cosine horizon40；warmup2、patience6不变 | ET_LONG_F01 |

前四项与基准相同30轮预算；LONG按请求研究延长日程，是**明确的预算例外**，不能称为等Epoch对照。所有任务同为最多12 GPU小时的保护上限、seed42、384、同Fold、同样本覆盖、同优化器更新定义；不因结果临时改参数。若原early stopping先触发，LONG不会强迫跑满40轮。没有新ASL、Mixup或架构实验，不重复Phase32八项。

## 一次配置与启动

使用发布消息中的完整commit，在独立克隆目录checkout `friend-gpu-batch`。复用朋友既有Python3.12/3.13、torch2.10.0/torchvision0.25.0 CUDA环境；依赖同 `requirements-phase32.txt`，不自动改驱动或下载权重。已有环境中安装本仓库可用 `python -m pip install -e . --no-deps`。

复制 `configs/e08_tune_server.example.yaml` 为 `configs/e08_tune_server.local.yaml`，填写授权绝对路径和GPU资源：

- data_root：官方cxr_first.zip，或保留官方内部布局的train.zip/valid.zip/test.zip目录；不支持任意未合并二进制分片。
- pretrained_root：包含hub/checkpoints/effnetv2_s-dd5fe13b.pth；SHA由配置强校验，不自动下载。
- protocol_root：原冻结patient_folds.csv、study_folds.csv、image_folds.csv，不重新划分。
- baseline_results_root：历史phase17_batch，需N17五折的原始Val/OOF分析、raw/EMA指标、配置、训练历史、manifest；最终E17对照还需D/C/S五折原始OOF文件。不能用只有指标没有概率的副本替代。
- output_root：全新目录，末级必须是e08_tuning，不能指向Phase17/32输出。
- gpu_ids/max_concurrent：朋友授权的空闲GPU，最多8卡一任务一卡。
- gpu_hours：本批总GPU小时硬上限；extension_gpu_hours：其中允许补折的子上限，填0即不扩展。两者不能留空，子上限不能超过总上限。

唯一推荐命令：

```bash
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml
```

程序先检查空闲设备、CPU/RAM/磁盘、数据患者隔离、基线哈希、配置与代码哈希，再逐项真实短步前反向、AMP、checkpoint恢复、Val/无标签Test短推理。全部通过才开始Pilot；扩展前对入选的新Fold再次预检。不要求人工二次启动。旧任务正在占卡时阻止启动，不能抢占朋友其他作业。

短测及正式训练计入预算；每任务启动前预留12小时，真实结束归还未用预留。5个Pilot上限60h+1.25h短测；最多补8折96h+2h短测，合计159.25h是**保护上限，不是预估耗时**。朋友可授权更少，剩余任务PENDING。N17历史约2.3h/Fold仅作背景；实际新任务时间/显存待测。推荐120GiB空闲输出空间（不含数据），每并发任务4 CPU与8GiB可用RAM；不自动减少batch或更改训练参数来规避OOM。

```bash
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action status
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action stop
python scripts/e08_tune_run.py --config configs/e08_tune_server.local.yaml --action resume
```

stop为排空：当前模型完成，不再领新任务。意外中断后恢复完整Epoch状态；不重复DONE任务。死父进程留下的GPU小时预留保守计入一次。失败/无恢复checkpoint时保留文件并报错，OOM/数值异常不自动改配置。预检失败阻止训练，日志保留；同一命令支持已通过预检及正式训练断点复用。

## Pilot筛选与扩展门禁

必须五项Pilot均完成评价，否则扩展BLOCKED。仅看EMA相对同Fold冻结N17 EMA：OOF Macro-AUC差值≥.002、患者级2000次有效Bootstrap的95%差值区间下界>0、任何类别AUC下降不超过.02。无效单类抽样计数并跳过；seed42。按OOF差值降序、区间下界降序、变体ID升序确定并列；最多2项。

选择结果及来源metrics哈希冻结在manifests/selection.json。仅入选变体生成Fold2–5队列，最多8个额外训练；没有过门槛则停止。25份配置是五种方法×五折的**预备模板**，不是25项默认训练；实际最多13项。不得手动挑选未过门槛的模板直接运行。

完成后按对应未见患者模型拼接完整OOF，核验18513 Study/11055患者；对N17 EMA和固定E17作配对Bootstrap。另输出Fold2–5的比较，区分Fold1选择偏差。所有“完整OOF”只说明来源隔离和覆盖完整，不能消除已重复使用Fold1/官方Val及既往OOF选择的偏差，不能称为全新独立测试。

## 产物、同步与替换E08条件

`e08_tuning/pilot/experiments/ET_*/`及`expansion/experiments/ET_*/`分别保存raw/EMA纯权重及哈希、完整last_training.pt恢复状态、配置、每轮train_history.csv/metrics.jsonl、最佳Epoch、Val/OOF概率、10类AUC/AP、显存时间、manifest、DONE/FAILED。每项结束即更新所属阶段reports/experiment_summary.csv。

顶层oof和validation分别保存完整OOF、五折等权Val；reports/five_fold_summary.csv、five_fold_analysis.json含完整及Fold2–5区间。始终区分单折PARTIAL_OOF与完整结果，不输出Test AUC。

默认结果不上传。朋友若已授权，将upload.enabled设true，并配置独立checkout和同一私有仓库的 `results/e08-tune` 分支，复用已登录gh权限。完成模型后增量上传小型结果到 `results/e08_tuning/pilot`或`expansion`，最终追加聚合分析；不传原始数据、模型权重、凭据，不改friend-gpu-batch代码分支。上传冲突记录错误，不丢训练结果。结束后从该独立结果分支同步；权重另行授权交付。不压缩打包。

值得替换E08组成部分至少需：完整对齐OOF及未用于本轮筛选的折有支持、逐类无明显损害、保持0.5/0.5公式后的Val复核、全部实际模型≤500MB、T4完整端到端≤100ms实测。五个EfficientNet模型加E03约605MB，不能直接当作合规E08替代；当前程序记录其大小下界风险，不自动导出Submission。若需最终单个全Train重训，须另行冻结并批准协议，本批不暗中增加任务。4090研究延迟不是T4合规证明。
