# Mixup 与灰度增强：独立患者确认实验

2026-11-10：Pilot已验收，仅Mixup通过。已授权的下一阶段为Fold3–5各BASE+MIXUP共6项，服务器操作见 [MIXUP_FOLDS3_5.md](MIXUP_FOLDS3_5.md)。NO_PHOTO不扩展。

本批是已有信号的受控确认，不是保证提分的新模型。保留 E08 官方 77.75 为基准。没有修改当前 Windows 4060 训练的文件、进程或配置。

## 首批实际训练什么

|队列任务|实际模型训练|唯一变化|外层评估|
|---|---|---|---|
|MIXUP_F02|BASE → MIXUP|Mixup alpha=.2、每微批一次、固定概率1|Fold2|
|NO_PHOTO_F03|BASE → NO_PHOTO|关闭亮度/对比度扰动；保留相同几何增强及随机数抽样顺序|Fold3|

共 **4 次训练、2 个成对任务**。默认每对独占一张 GPU，BASE 完成后接着挑战模型，两对可并行；最多用2张卡。没有为了占满8卡增加实验。现有主机 GPU 锁与 Phase32/35/41/42/StrongKD 互斥；其他队列运行时本批拒绝抢卡。队列预算、失败隔离与状态表复用 Phase42。

本地正在运行 MIXUP_F02。同名服务器任务是备份复现，不能把两次结果当成独立患者或挑最好的一次。若朋友只补充新方向，在配置中设置 `pilot_pairs: [NO_PHOTO_F03]`、`gpu_ids: [0]`、`max_concurrent: 1`。

### 证据与假设

- Mixup：历史 Phase32 Fold1 EMA AUC .8089316745，对 N17 EMA +.0047420160，95% CI [.0012255349,.0082023691]；十类点估计均正。
- NO_PHOTO：历史同折 EMA AUC .8087119926，+ .0045223341，95% CI [.0011016241,.0081896037]；最差类别 -.0053403233。假设轻微光度增强可能破坏部分胸片灰度线索，需要受控确认，尚未证明。
- 历史两个实验都用官方 Val 早停，且参与过方向选择。本次重新训练同条件 BASE，在训练患者内部早停；不能直接拿旧 N17 分数作严格对照。
- 不追加 E04/大模型/蒸馏搜索：E04 未通过旧19项多重比较门槛，强教师蒸馏未稳定胜过原教师。优先集中资源验证这两个现有线索。

证据来源：既有 `phase36_release_acceptance/pilot_comparison.csv`，Phase32结果提交 `9fd6afc68b904ac4cf7f319053f09023507ca5ee` 的 `P32_NS_MIXUP`、`P32_NS_PHOTO`。这些是历史探索成绩，当前批次没有生成任何训练成绩。

## 朋友首次设置

在**独立代码克隆**中拉取 `friend-gpu-batch`，检出本次交付的完整 commit。不要在正在运行的旧队列目录中 pull 或切分支。要求 Python3.12、PyTorch2.10.0、torchvision0.25.0、CUDA可用。

```bash
git clone --branch friend-gpu-batch --single-branch git@github.com:Zoey-711/cxr-multilabel-challenge.git cxr-confirmation
cd cxr-confirmation
# 用交付消息中的真实SHA检出：git checkout <commit>
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
cp configs/confirmation_server.example.yaml configs/confirmation_server.local.yaml
```

填写 `configs/confirmation_server.local.yaml`：

1. `data_root`：已获授权的 `cxr_first.zip` 完整绝对路径。
2. `pretrained_root`：目录下必须存在 `hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth`，SHA-256 `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`。不会自动下载或用医学预训练替代。
3. `protocol_root`：已有患者五折索引目录。仓库自带的 `outputs/analysis/phase17_patient_folds/` 可用，填写其服务器绝对路径，三个CSV逐个验证原哈希。
4. `output_root`：新的独立输出目录；不得填写旧Phase17/32/35/41/42目录。
5. 授权 GPU 编号、并发数和 `pair_gpu_hours`、`gpu_hours`。例如每对20 GPU小时、两对总40 GPU小时，是预算上限示例，不是测速估计；确认后自行填写数值。每对预算包含真实数据预检、两个模型训练及评估。

锁定Holdout索引已经放入 `protocol_assets/confirmation_patient_split.csv`，保持原始字节与SHA；仅含匿名患者ID及split，不含影像或疾病标签。它用于**排除**既有Holdout，不能重新用于选择模型。

先 CPU 审计，然后一条命令正式运行：

```bash
python scripts/confirmation_run.py --config configs/confirmation_server.local.yaml --action prepare
python scripts/confirmation_run.py --config configs/confirmation_server.local.yaml --action run
```

`run`自动核查 GPU空闲、24GB4090、CPU/RAM/磁盘预算、代码完整性和全部来源哈希。每对先完成真实图片CUDA短步、EMA和恢复检查，再启动训练；预检不通过不启动该对训练。另一对不会因它失败被取消。每项完成即保存结果；两项都完成后才读取外层评估指标。

## 冻结训练及验证协议

384、ImageNet EfficientNetV2-S、BCE、batch8×累积2、AdamW(backbone1e-4/head3e-4，wd1e-4)、warmup2+cosine30、FP16、clip1、EMA .999含buffers、seed42、patience6、严格改善才换最佳checkpoint。两组初始化哈希必须相同，BN按同样微批量更新。

NO_PHOTO把brightness/contrast幅度设为0，但仍抽取原来的Bernoulli和两个幅度随机数；光度变换为恒等，几何增强和后续随机层不会因少抽随机数而改变。CPU测试核验其恒等像素与随机流。Mixup使用独立NumPy/Torch RNG。

剔除旧Holdout后，固定外层Fold。其余患者按 SHA256(`MIXUP_CONFIRM_V1|Subject_id`)排序，前ceil(10%)作为内层早停集。每组患者互斥；外层患者不进入训练、BN、损失或早停。当前实现只读取Train，不读取官方Val/Test。

Fold2排除Holdout后的已审计覆盖：训练7146患者/11914 Study；内层795/1365；外层2008/3388。其他折按真实索引生成。**结果是PARTIAL_OOF，绝不是18513 Study完整OOF**。这些患者历史上曾用于其他研究，也不能称全新盲测。

GradScaler先unscale，有限梯度才clip；成功optimizer step才推进LR和EMA。连续3次或累计超过20次跳步即失败，事件逐个保存。每个epoch保存模型/EMA/optimizer/scheduler/scaler/早停/完整RNG；新增最佳checkpoint使用带epoch的不可变文件，恢复时核验last状态指向的最佳文件，防止中断造成最佳记录错配。

## 输出、恢复与查看

每对位于 `output_root/experiments/<任务ID>/`，含 `BASE/` 和 `MIXUP/` 或 `NO_PHOTO/`。各模型有最佳权重、完整恢复状态、配置、逐轮历史、inner AUC、OOF预测和指标。外层评估完成后有2000次患者配对Bootstrap、逐类支持数/差值、`final_decision.md`、SHA清单和DONE凭证。

```bash
python scripts/confirmation_run.py --config configs/confirmation_server.local.yaml --action status
```

同一命令再次`run`会核验已完成结果并跳过；中断任务检查父子PID后从完整epoch边界恢复。明确FAILED的任务不会自动重试；排除故障、核实剩余预算后，用同一命令将`--action run`改为`--action retry-failed`，历史attempt日志仍保留。非有限checkpoint会被拒绝，不会自动改变微批量或精度。不要删除仍在运行任务的锁。达到预算超时会终止该任务并保留上一完整epoch；不要把运行中的临时文件当成恢复文件。

## 固定继续/停止规则

初筛两条路线各自要求外层delta≥.001，最差类delta≥-.01。报告95% CI，但初筛通过只标PROMISING_PILOT。Fold2/Fold3已用于各自初筛，不能当后续独立确认折。

后续必须新配置、新输出目录、单独授权预算，**不会自动启动**：

- `stage: mixup_confirmation`：Fold3/4/5各BASE+MIXUP，读取`pilot_results_root`下MIXUP_F02通过凭证。合并确认折delta≥.002、95%CI下界>0、至少2/3折正向、最差类≥-.01。保持已批准Mixup主方案标准。
- `stage: no_photo_confirmation`：Fold4/5各BASE+NO_PHOTO，读取NO_PHOTO_F03通过凭证。合并delta≥.002、**97.5%CI下界>0**、两折均正向、最差类≥-.01；同时报告95%区间。第二条路线采用更严格区间，并披露两方向筛选偏差。

若两个确认阶段都运行，会分别训练Fold4/5 BASE；这是明确计入预算的各自同条件对照，不宣称为额外独立患者。若要节省重复BASE，须先实现并审计严格来源复用，当前不会擅自串用检查点。运行后 `--action analyze` 拼接每位患者自己的折外预测，不平均不同折OOF，不把单折AUC取平均代替合并AUC。

即使确认通过，仍需下一阶段全Train固定预算对照及E08固定组成替换评估；不能承诺官方超过77.75。无新增推理结构，权重规模与EfficientNetV2-S一致，但没有新模型大小或T4实测成绩。

## 增量结果交付

每对完成即在 `output_root/exports/<任务ID>/` 自动导出经过哈希核验的轻量结果。仅含预测、指标、配置摘要和来源；服务器路径脱敏，不含原始图像、疾病标签、checkpoint或凭据。原始与脱敏文件哈希分别记录。权重留在服务器，需要时再单独授权传输。

若朋友已有私有GitHub写权限，可填 `upload.checkout` 为独立结果克隆目录，确认授权后设 `upload.enabled: true`，每完成一对即可运行：

```bash
python scripts/confirmation_run.py --config configs/confirmation_server.local.yaml --action upload
```

目标固定为 `results/regularization-confirmation-20261108` 分支，目录 `results/regularization_confirmation/<来源identity>/<任务ID>/`。校验仓库Private与写权限、禁止覆盖不同历史文件、禁止强制push。默认不上传，训练代码分支不接收实验产物。

本地收到结果后可从该分支同步CSV/JSON；按export_manifest核验所有轻量文件，源权重仅记录SHA并标记WEIGHTS_NOT_DOWNLOADED。无官方Test预测，无Submission，无自动榜单提交。

## 本地测试边界

Windows只运行CPU测试、导入、语法与真实元数据审计，不占用4060训练GPU。原本地Mixup短步CUDA测试已通过；本次移植版的Linux4090真实数据预检由朋友执行，不能把本地CPU通过当作服务器训练通过。

```bash
python -m unittest discover -s tests -p 'test_regularization_confirmation.py' -v
```
