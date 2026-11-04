# 单模型蒸馏 v2：朋友服务器运行说明

本队列独立于 Phase42。默认3项，可选第二组完整配对复现后共6项。不重建教师，不生成Submission。本地只做CPU测试，真实Linux4090预检仍待服务器执行。

## 固定实验

|任务|监督信号|种子|
|---|---|---:|
|STUDENT_BASE|纯真实标签BCE，不消费教师概率|42|
|SKD_CONTROL|0.5真实BCE + 0.5原教师软标签BCE|42|
|SKD_COMPLEMENT|0.5真实BCE + 0.5强教师软标签BCE|42|

原教师：`0.25 D17 + 0.25 C17 + 0.50 S17`。强教师：`0.125 D17 + 0.125 C17 + 0.25 S17 + 0.50 N17_EMA`。固定比例，不搜索。

可选复现增加三项 `_SEED2026`：模型初始化和数据随机流均改为固定seed2026，不事后挑最佳种子。

共同设置：ImageNet EfficientNetV2-S、384输入、微批8×累积2、worker0、AdamW骨干1e-4/分类头3e-4、weight decay1e-4、2轮warmup、固定4轮、FP16 AMP、EMA .999含buffers、conservative_cxr_v1增强、黑边填充和Study Mean。同种子三个模型初始化和数据顺序保持一致。软目标按Subject/Study显式映射，软损失按完整Train每Study图像数倒数加权；不是直接对Study Mean训练。

全Train学生、859 Study官方Val开发评估，**学生OOF=N/A**。官方Val已重复用于模型选择，Bootstrap仅为探索性证据。当前普通教师OOF不能直接切成无泄漏外层Fold1学生目标。

## 取得代码与环境

不要更新正在训练的代码目录，使用独立克隆：

```bash
git clone --branch friend-gpu-batch --single-branch git@github.com:Zoey-711/cxr-multilabel-challenge.git cxr-strong-kd-v2
cd cxr-strong-kd-v2
git checkout <交付消息中的完整commit>
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase32.txt
cp configs/strong_kd_server.example.yaml configs/strong_kd_server.local.yaml
```

可以复用此前验证可用的torch2.10/torchvision0.25 CUDA环境。不要升级正在训练的环境。

## 必须填写四个真实绝对路径

- `teacher_root`：原Phase17 `results/phase17_batch`，其下直接有`manifests/`和`experiments/`。来源commit `cbfc5997438f371086cd1fbe232890590cd67ffb`。`configs/strong_kd_sources.json`锁定106个轻量文件，包括20任务OOF、训练患者索引、DONE、manifest、配置及划分。无需20个教师权重；不能仅提供汇总表。
- `data_root`：授权完整`cxr_first.zip`或含`train.zip`、`valid.zip`的原布局目录，不需要Test。
- `pretrained_root`：其下有`hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth`，固定SHA `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`。
- `output_root`：新独立目录，例如`/work/outputs/strong_teacher_kd_v2`。不能复用旧两任务目录；代码/配置身份改变会拒绝恢复。

先做CPU审计：

```bash
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action audit
```

默认三项资源授权示例（朋友明确同意后填写）：

```yaml
gpu_ids: [0]
max_concurrent: 1
authorize_training: true
gpu_hours: 38
preflight_gpu_hours: 2
replicate_seed_enabled: false
replicate_gpu_hours: 0
```

若预先决定运行第二组三项，设置 `replicate_seed_enabled: true`、`replicate_gpu_hours: 36`、`gpu_hours: 74`。每任务硬超时预留12GPU小时，三项36/六项72加预检2小时。**38/74是保守预算上界，不是实测耗时预测。**

最多2张4090，默认单卡顺序运行。双卡可用`gpu_ids: [0, 1]`和`max_concurrent: 2`。至少15GiB空闲磁盘，每并发任务至少6GiB可用RAM及2CPU核，默认显存上限20GiB。共用主机GPU锁，不与Phase42等队列争抢GPU。

## 一条启动命令

```bash
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action run
```

先审计来源/患者划分/哈希，再检查空闲4090及资源预算；每项真实数据CUDA短步通过optimizer/EMA更新、完整checkpoint保存恢复和Val前向后，才启动正式训练并自动分析。依赖安装不等于CUDA通过，不静默回退CPU或更改batch/精度。

```bash
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action status
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action stop
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action resume
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action analyze
```

stop排空当前任务后不派发。恢复最近完整Epoch，包括模型/EMA、optimizer/scheduler、GradScaler、RNG、数据加载器状态；完成任务不重训。单轮20次或连续4次AMP跳步即失败，不无限重试。预检异常中断保留PID/预算账本，需要核实，不删除账本绕过门禁。运行中不能更改种子开关或预算配置；新增批次使用独立输出。

## 自动报告与冻结判断规则

每项`experiments/<ID>/`立即保存配置、训练历史、日志、raw/EMA推理权重、完整恢复状态、Val概率、逐类别指标、模型大小、manifest、DONE哈希。固定最后第4轮raw是主要比较，EMA是预声明次要比较，不用Val挑epoch。

队列结束自动生成，也可单独`--action analyze`：

- `reports/student_metrics.csv`：每模型raw/EMA Macro-AUC。
- `reports/student_per_class.csv`：10类AUC和阳性Study数量。
- `reports/bootstrap_comparisons.csv`：每种子独立比较CONTROL−BASE、COMPLEMENT−BASE、COMPLEMENT−CONTROL，2000次患者级配对Bootstrap（seed5417）。
- `reports/student_comparison.json`：指标、Bootstrap有效/无效抽样记录、缺失任务及决策。
- `reports/final_decision.md`：中文结果表。

强教师学生必须同时超过BASE与CONTROL：raw增量均≥0.002、95%CI下界均>0、任一类别下降不超过0.01。启用复现时两组均须通过。缺任务仅报告WAIT，不挑最高分种子或EMA。多个比较、重复使用Val和仅一/两个种子的限制都必须保留，即使通过也只建议考虑外层教师重建，不能宣称独立泛化确认。

无泄漏外层确认需要教师完全排除外层验证患者，例如排除Fold1后在其余四折重建D/C/S/N×4内折共16教师。本入口不执行这些训练。没有收益则停止教师重建。不生成Test/Submission，T4_UNVERIFIED。

## 回传

本入口不自动上传。朋友通过已有授权私有方式交付独立输出，建议新位置`results/strong-teacher-kd-v2`（本代码不创建分支，不保证已存在）。优先报告、配置、CSV与manifest，权重另传；不要上传原始图像、标签、凭据或缓存，不覆盖Phase17/Phase42。完整`analyze`会核验DONE全部文件，缺权重不能声称完整验收通过。可先阅读小体积报告，再按需取得完整产物。
