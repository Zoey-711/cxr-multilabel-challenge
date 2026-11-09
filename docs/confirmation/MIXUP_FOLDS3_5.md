# 只运行Mixup Fold3–5确认：2026-11-10

已批准：Fold3、4、5各BASE+MIXUP，共6次新训练。停止NO_PHOTO方向，不重训Fold2。

通过依据：Release regularization-confirmation-pilot-full-v1，结果分支commit bfe369832d24f94fd50310726dd8eca464ddc64a。外层Fold2 BASE=0.8003662971、MIXUP=0.8041487705，delta=0.0037824734，患者Bootstrap95%CI=[0.0003789933,0.0070080348]，最差类delta=-0.0016968519。仍为PROMISING_PILOT。

## 朋友执行

在独立、干净的代码目录检出本次交付commit，使用原Python环境及此前成功的 `configs/confirmation_server.local.yaml`。原配置的output_root必须仍指向服务器完整Pilot目录，其中包含 `experiments/MIXUP_F02/DONE.json` 与原始manifest。不要用GitHub脱敏导出目录替代它，导出目录没有原始DONE凭证。

```bash
# 在该代码目录执行。仅选择空闲且已获授权的GPU；不足3张时可只写0或0 1。
python scripts/prepare_mixup_confirmation.py --config configs/confirmation_server.local.yaml --gpu-ids 0 1 2
python scripts/confirmation_run.py --config configs/mixup_folds3_5.local.yaml --action prepare
python scripts/confirmation_run.py --config configs/mixup_folds3_5.local.yaml --action run
python scripts/confirmation_run.py --config configs/mixup_folds3_5.local.yaml --action analyze
```

`prepare_mixup_confirmation.py`只生成配置，不启动训练；新配置已存在时拒绝覆盖。自动复用数据、预训练、冻结划分、Holdout排除清单，输出改为原目录旁的`<原目录名>_mixup_folds3_5`，引用原始Pilot通过凭证。队列只包含MIXUP_F03/F04/F05，每对BASE完成后运行MIXUP。GPU不足时统一排队。

每对预算沿用原配置pair_gpu_hours，总预算设为其3倍。原配置若为20小时/对，则总预算60 GPU小时；这是硬上限而非耗时保证。此次4090已完成的Mixup对约15.78 GPU小时，3对按同吞吐粗略约47.35 GPU小时，3卡并行约16小时，早停和负载会改变时间。若需要覆盖30epoch最坏情况，请在启动前明确设置更高预算，例如`--pair-gpu-hours 48`（总144 GPU小时）；不可将该数字误认为必需实际耗时。预算超时按原队列机制终止对应任务并保留上一完整epoch。

保持现有训练逻辑与全部冻结超参数：384、batch8×累积2、ImageNet初始化、seed42、EMA .999、Mixup alpha .2、最多30epoch、patience6、内层早停。每折先验证数据隔离与CUDA短程；已有GPU任务/锁存在则拒绝抢占。

恢复与查看：

```bash
python scripts/confirmation_run.py --config configs/mixup_folds3_5.local.yaml --action status
# 未标FAILED的中断任务：再次run，原有恢复检查生效。
# FAILED任务：先查queue_logs，排除原因后使用--action retry-failed。
```

## 确认标准与交付

训练完成后`analyze`按患者所属Fold拼接Fold3–5，绝不平均OOF或拼旧协议Fold1。主结果只用这三个确认折：delta>=.002、患者Bootstrap95%CI下界>0、至少2/3折正向、最差类别delta>=-.01。失败即停止，不改变Mixup参数。结果是排除Holdout的PARTIAL_OOF，不是18513 Study完整五折；官方Val/Test不参与本阶段。

每对完成自动生成轻量exports。上传仍用原私有结果分支`results/regularization-confirmation-20261108`，目录`results/regularization_confirmation/<新identity>/MIXUP_F03|F04|F05/`，不会覆盖Pilot。新配置默认upload.enabled=false；确认朋友拥有写权限后，将新配置此项设true再执行：

```bash
python scripts/confirmation_run.py --config configs/mixup_folds3_5.local.yaml --action upload
```

注意：输出identity包含服务器配置。应在第一次prepare之前固定upload.enabled及上传路径，运行后不要编辑配置；如需稍后开启上传，应使用独立仅供upload动作的配置副本，不用它继续run或prepare。上传使用独立干净结果克隆，保留已有历史。最终同时交付reports/confirmation.json与reports/confirmation_per_class.csv；现有upload导出以每对结果为主，这两个合并报告需要另外作为Release轻量附件交付。最佳权重按需放新Release，记录SHA，不上传原图或原始标签。

本地Windows只做CPU测试；本说明不表示朋友服务器已启动。当前本地4060备份运行不受影响。不得择优选服务器/本地同折结果当独立确认。
