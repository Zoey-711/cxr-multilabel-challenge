# Phase42：19项候选 Pilot 与最多16项跨折确认

本批保留17个结构与2个训练策略，详细固定定义、去重原因及CPU计算量在 [实验清单](experiment_catalog.md)。Phase42最初8项未运行；本文件对应扩展版唯一计划。历史数据和模型均不改变，训练只在朋友授权的Linux服务器执行。

## BASE与启动依赖

必须先取得Phase41同协议BASE的Fold2完整产物。微批4×累积4、worker0与BatchNorm行为沿用Phase39/41；N17原微批8×累积2不能当严格对照。**不增加或重复Phase41训练**，缺少BASE即阻止正式启动。跨折确认进一步要求Phase41 BASE Fold1–5全部完成；Fold1为已冻结Phase39 BASE。启动前填入原Phase41输出路径，包含`experiments/P41_BASE_F02/`、`preflight/REMOTE_PREFLIGHT.json`，确认阶段还需其余BASE及`reused_fold1/P39_BASE_F1/`。原始Git来源声明和已核验SHA单独记录，不以文件夹名称冒充来源验证。

使用原有兼容环境（`requirements-phase32.txt`，torch2.10/torchvision0.25），在独立克隆目录检出交付消息给出的完整commit。不要更新正在训练的代码目录。数据通过赛事允许方式单独取得：`data_root`支持完整`cxr_first.zip`或含`train.zip`、`valid.zip`的原布局目录。本批不读取隐藏Test标签，也不生成Submission。

```bash
git clone --branch friend-gpu-batch --single-branch git@github.com:Zoey-711/cxr-multilabel-challenge.git cxr-phase42
cd cxr-phase42
git checkout <交付消息中的完整commit>
cp configs/phase42_server.example.yaml configs/phase42_server.local.yaml
```

填写绝对路径：赛事数据、合法ImageNet缓存、冻结五折CSV、Phase41结果、独立`output_root`。ImageNet缓存文件为`hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth`，完整SHA已固定。填授权GPU编号、最多8并发、单任务显存上限（默认20GiB）、磁盘/RAM限制及预算。总预算必须覆盖`pilot_gpu_hours + preflight_gpu_hours + confirmation.gpu_hours`；预算空缺拒绝运行。每个正式任务保守预留48GPU小时，实际按进程用时记账；异常失联按一次完整预留额度扣账。48小时是超时上界，不是4090实测估计。8项同时派发需至少384小时可用预留，较小预算自动降低实际并发；不足48小时不派发新任务。19项实际耗时待远程短测，不能沿用4060速度推断。

## 一条启动命令

```bash
python scripts/phase42_run.py --config configs/phase42_server.local.yaml
```

入口依次执行：合法文件/患者划分/BASE校验 → Linux资源及空闲4090检查 → 19种模型真实微批CUDA优化器步进、重建模型并加载完整恢复状态、恢复后再次优化、OOF/Val短推理 → 按1→2→4→授权并发数逐级检查 → 19项Pilot动态排队 → CPU筛选建议。

正式训练每次用Fold2的其余四折训练，EMA Fold2 Study Macro-AUC选Epoch；Val只在训练结束后导出，不能参与结构筛选。失败任务保留checkpoint和原因；OOM、数值问题不自动改batch或超参数。全局GPU锁与Phase32/35/41共用，两套队列不能各占8张卡。

默认只跑Pilot。若朋友在开始前明确设置`confirmation.enabled: true`并填入独立确认预算，同一命令在Pilot全部到达DONE/FAILED/BLOCKED后，按冻结规则最多选4个不同机制，再自动补各自Fold1、3、4、5，共最多16项；已有Fold2绝不重训。BASE缺失时写`CONFIRMATION_BLOCKED.json`，不会偷用另一协议模型。预算不足保留PENDING。第三阶段仅生成最多2个建议，不自动执行未固定协议的全Train/学生训练。

## 冻结筛选与统计边界

Pilot要求相对同折BASE的Macro-AUC增量≥0.002、2000次患者配对Bootstrap 95%区间下界>0、最差类别降幅不超过0.01、MAC比≤1.35、推理权重+配置≤500000000字节，并通过19候选家族的近似中心化Bootstrap单侧p值Holm门限0.05。每种机制最多1个，最多4个；按增量、MAC比、ID确定并列顺序。失败候选仍计入19个比较的家族大小，不因为缺失而缩小惩罚。该p值与区间属于探索性统计，不是独立确认性显著性证明。

确认阶段同时报告完整五折和排除Fold2后的OOF，不平均单折AUC。逐患者核验来源，完整覆盖18513 Study/11055患者才称完整OOF。后四折增量≥0.001、区间下界>0、最差类别降幅不超过0.01才进入最多2个后续建议；最多4次确认比较、每折早停以及历史队列重复研究仍有选择偏差。Fold2从未用于本轮参数再调整，但不是从未被历史研究接触的新数据。Val五折先均值概率再算AUC；E08仅作0.8116686461标量参考，未加载其CSV时不声称做了配对E08比较。

## 查看、停止、恢复、回传

```bash
python scripts/phase42_run.py --config configs/phase42_server.local.yaml --action status
python scripts/phase42_run.py --config configs/phase42_server.local.yaml --action stop
python scripts/phase42_run.py --config configs/phase42_server.local.yaml --action resume
python scripts/phase42_run.py --config configs/phase42_server.local.yaml --action upload
```

停止采用排空方式，当前任务正常结束后不派发新任务。正常意外中断后重复启动会校验DONE、按完整Epoch恢复；显式STOP后用resume。若预检进程本身被外部杀死，保留running账本并拒绝自动清零，须先核实PID与消耗；不要删除账本绕过预算。

每项产物在`output_root/experiments/P42_<ID>_Fxx/`：EMA最佳权重和SHA、对应原始权重、完整恢复checkpoint、配置、历史、OOF/Val概率、逐类指标、环境、manifest、计算量、参考端到端延迟、DONE/FAILED。官方Test不在本批交付范围。`reports/pilot_comparison.csv`、`selection.json`、`confirmation_analysis.json`、`analysis/*`、`oof/*_fivefold.csv`、`validation/*_fivefold.csv`直接保存在分类目录。没有ZIP/TAR打包。

需要自动回传时，朋友明确设置`upload.enabled: true`并配置独立上传checkout，服务器须已通过`gh auth login`与Git认证且有Private仓库写权限。轻量产物进入私有分支`results/phase42-pilots`下`results/phase42_batch/`；EMA权重逐文件进入Release `phase42-pilots-v1`。首次使用前检查名称冲突；不同哈希的同名资产拒绝覆盖。原始图像、原始标签、恢复checkpoint和凭据均不上传。每项完成即本地持久化并尝试增量回传，失败保留`UPLOAD_FAILED.json`，可单独重试上传。

Windows独立接收（首次clone，以后pull --ff-only）：

```powershell
git clone --single-branch --branch results/phase42-pilots git@github.com:Zoey-711/cxr-multilabel-challenge.git "E:\计挑赛\cxr-friend-results\phase42-results"
git -C "E:\计挑赛\cxr-friend-results\phase42-results" pull --ff-only
```

记录该结果commit，先核验`export_manifest.json`文件哈希，再验每项DONE、Fold来源、完整OOF覆盖及`confirmation_analysis.json`。最佳权重在`binary_release_manifest.json`列出资产名/SHA，按需下载，不自动下载全部权重。参考BASE轻量证据在`reference_baselines/`，无原图、无原始标签。

本地复算时使用独立服务器配置副本，`output_root`指向下载的`results/phase42_batch`，`phase41_results_root`指向其中`reference_baselines`，`data_root`和`protocol_root`指向本地合法Train/Val与冻结索引；执行`--action analyze`只做CPU分析。完整结果包含`queue_pilot.json`供状态复核；模型权重可保持未下载，指标分析明确跳过缺失二进制的哈希核查，不能把这种状态称为权重验收通过。

## 本地测试与未验证事项

CPU测试覆盖全部19结构的前向/反向/EMA/纯权重重载、同初始化、损失、患者隔离、预算、失败隔离、恢复导出与完整OOF汇总。真实Linux4090运行和显存仍需服务器预检。机器上检测到GPU占用，本轮未运行本地CUDA训练；已实现远程真实CUDA预检，不冒充实测。研究测速包含读图到CSV关闭，5张预热、100张×3轮；并行负载和ZIP打开范围均记录，不声称是官方T4成绩。全部候选保持`T4_UNVERIFIED`。

本次连同Phase41/35回归共 **58 passed、0 failed、0 skipped**；证据、测试命令及文件SHA见[local_acceptance.json](local_acceptance.json)。这些都是CPU测试，不能将未执行的CUDA实机检查计为通过。

BASE??????Phase41???Fold2??EMA??????????????105?Val???5????100??3?????????????BASE??????????????????????????????????
