# Historical v1 documentation; not current launch instructions

# 更强 OOF 教师 → 单模型学生：受控蒸馏实验

代码入口为 `scripts/strong_kd_run.py`，独立于 Phase42 的 19 项结构实验。
默认只进行 CPU 审计；本地没有启动正式训练。

## 已有证据与固定方案

来源固定为 `Zoey-711/cxr-multilabel-challenge` 结果 commit
`cbfc5997438f371086cd1fbe232890590cd67ffb` 的 `results/phase17_batch`。
`configs/strong_kd_sources.json` 锁定106个轻量文件的 SHA-256，包括标签索引、五折划分、20个教师任务的 DONE、配置、训练患者索引、manifest 和 OOF。教师权重只核对 manifest/DONE 中的哈希对应关系，不下载教师权重，也不冒充完成二进制权重验收。

2026-11-04 本地 CPU 独立复算，18,513 Study、11,055患者：

| 教师 | 完整教师 OOF Macro-AUC |
|---|---:|
| 原 H17 教师：0.25 D17 + 0.25 C17 + 0.50 S17 | 0.8043439291 |
| N17 EMA | 0.8076721228 |
| 固定 0.5 E17 + 0.5 N17 EMA | 0.8124035273 |

新教师相对原教师增量为 **0.0080595981**。2,000次患者配对 Bootstrap（seed=5417）95% CI 为 **[0.0068954109, 0.0092298172]**，无无效抽样。十类 AUC 都有提高。这是教师集成互补性的证据，尚未证明学生可以保留该收益。

只训练两个同条件学生，不搜索融合比例或蒸馏参数：

| 任务 | 软标签公式 |
|---|---|
| SKD_CONTROL | 0.25 D17 + 0.25 C17 + 0.50 S17 |
| SKD_COMPLEMENT | 0.125 D17 + 0.125 C17 + 0.25 S17 + 0.50 N17 EMA |

学生均为单个 ImageNet EfficientNetV2-S、384输入、seed42、micro-batch8×累积2、AdamW骨干1e-4/分类头3e-4、weight decay1e-4、2轮warmup、4轮固定训练、FP16 AMP、EMA0.999（含buffers）。增强沿用 conservative_cxr_v1，原始黑边填充和Study Mean不变。唯一组间变量是教师软标签。

损失为0.5真实标签BCE+0.5教师概率BCE，温度1。教师按Subject_id/Study_id显式映射到图像；软损失以完整Train中该Study图像数的倒数加权，并按每个梯度累积组分别归一化hard/soft分母，与H17的损失定义一致。该方法约束多图Study的权重，但不是直接对Study Mean输出训练，不能声称解决了视角粒度差异。

与历史H17的差异：本实验worker=0以保证Epoch边界RNG恢复；固定最后第4轮导出raw和EMA，不再用官方Val选最佳轮次。因此历史77.70学生不是本轮严格对照，必须使用新训练的SKD_CONTROL。raw最终轮为预先固定的主要比较，EMA为预先声明的次要比较。

## 验证范围与外层隔离

本实验是**全Train学生开发实验**。合法教师OOF覆盖每个训练患者，但学生使用全部Train训练，故学生OOF必须为 **N/A**；只有859 Study官方Val开发评估。官方Val此前反复参与教师及模型选择，Bootstrap仅为探索性比较，不是独立泛化证明。

不要把同一教师OOF按外层Fold1切开直接训练一个“无泄漏学生”：Fold2–5患者的OOF教师可能已见过外层Fold1。入口对 `mode` 非 `full_train_development` 直接标记 `BLOCKED_OUTER_TEACHERS`。

若后续要取得独立外层Fold1证据，需要先排除外层Fold1所有患者，在其余四折内建立内层OOF。D/C/S/N四种教师×四个内折，共16项新教师训练；每个教师的训练、统计量和checkpoint选择都必须排除外层验证患者，也不能用其软标签预测目标患者选择checkpoint。对照和新教师可共享这16项教师来源，然后训练两个外层学生。该重建目前只有方案，**本入口不执行或伪装实现这些额外训练**。

## 朋友服务器配置与命令

使用同仓库 `friend-gpu-batch` 交付commit及已有兼容环境 `requirements-phase32.txt`。
完整结果目录须在服务器上已可读；不需要下载20个教师权重。

```bash
cp configs/strong_kd_server.example.yaml configs/strong_kd_server.local.yaml
```

填写 `teacher_root`、数据、ImageNet缓存和独立输出路径。`teacher_root` 直接指向有 `manifests/`、`experiments/` 的 `phase17_batch`。
先进行CPU审计：

```bash
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action audit
```

实际训练前在配置中明确 `authorize_training: true`，填写 `gpu_hours`。默认仅1卡串行，最多授权2张4090并行。每个学生硬超时预留12GPU小时，加默认1小时预检，完整两任务应授权至少25GPU小时；这是预算上限，不是训练耗时预测。预算不足不派发新任务。与Phase42共用主机GPU队列锁，顺序执行，不争抢已占用GPU。

```bash
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action run
```

入口检查来源哈希、原始Train/Val映射、患者隔离、合法ImageNet哈希、空闲4090、磁盘/RAM/CPU和预算；真实数据短步完成梯度、optimizer/EMA更新、checkpoint重建恢复及Val前向后才启动正式任务。模型不在本地Windows训练。CUDA真实预检当前状态为 **PENDING_REMOTE**。

```bash
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action status
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action stop
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action resume
python scripts/strong_kd_run.py --config configs/strong_kd_server.local.yaml --action analyze
```

stop为排空停止，当前任务完成后停止派发。正常进程中断从最近完整Epoch恢复；AMP跳步记录到amp_events，单轮20次或连续4次即失败，不自动改精度。FAILED数值任务不会自动重试；最后成功Epoch保留。恢复checkpoint拒绝非有限参数、optimizer状态或不一致配置。预算失联按既有队列完整预留额保守计账，不清零已用额度。完成训练后的导出失败可从完成标记恢复导出，不重复训练。

## 结果与接收

`output_root/teachers/`为两份固定软标签及来源；`teacher_audit.json`记录完整审计和教师Bootstrap；`teacher_complementarity.csv`含逐类AUC、概率相关性、分歧、固定随机种子抽样的正负Study排序互补比例。排序抽样仅描述误差，不拟合教师权重。

每个 `experiments/SKD_*/` 保存：resolved_config、train_history、training.log、raw/EMA纯推理权重、完整恢复状态、raw/EMA Val概率、逐类指标、model_size、run_manifest、DONE及SHA清单。
`reports/student_comparison.json`对同一228患者进行2,000次raw/EMA配对Bootstrap。学生训练内预测不导出为OOF；不读取Test，不生成Submission。

继续标准在学生结果到达前固定：以最后第4轮 **raw** 学生为主要比较，`SKD_COMPLEMENT−SKD_CONTROL` 的Val Macro-AUC增量须≥0.002、患者Bootstrap区间下界>0，且十类中最差AUC降幅不超过0.01，才输出 `CONSIDER_NESTED_TEACHER_PILOT`；否则输出 `STOP_TEACHER_REBUILD`。即使通过也只是值得投入无泄漏外层验证的探索信号，不自动开始16项教师重建。EMA比较单独报告，不用其结果事后替换主要判断规则。

本入口不自动上传。结果由朋友经已授权的私有交付流程保存到独立位置，例如`results/strong-teacher-kd`；该分支尚未创建。优先传送报告、配置、CSV及manifest，学生权重按需单独交付，不能覆盖Phase17/Phase42。同步后先逐文件核验DONE中的SHA，运行`--action analyze`可以重算比较，但本版本完整验收需要学生权重也在本地；缺失时明确失败，不能伪称权重通过。

推理时只加载一个学生（raw或EMA之一），不加载教师或恢复checkpoint。文件大小按实际权重加配置记录；当前没有新学生真实产物，没有速度或AUC成绩。保持 **T4_UNVERIFIED**。开发结果即使改善也不自动扩大训练或提交排行榜。
