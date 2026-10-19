# Phase17-Batch：胸片多标签分类私有协作交付

本分支 `friend-gpu-batch` 用于朋友在自己的GPU服务器独立执行已批准的三骨干患者级五折实验。它不控制用户本地训练，也不需要远程登录朋友设备。

- [朋友三步操作、资源预算、停止恢复](README_friend.md)
- [结果包、校验、导入与离线复算](README_results.md)
- [冻结任务清单](configs/batch_plan.yaml)
- [历史成本估计](docs/batch17_costs.json)

## 基础与扩展档

基础15次：D17_F01–F05（B01）、C17_F01–F05（M02）、S17_F01–F05（T01）。官方Val选Epoch，当前Fold只生成OOF；训练输入384，原增强/BCE/优化器/学习率不变。ConvNeXt增加预先固定的EMA旁路，普通模型早停规则不变。

扩展：两个seed43模型默认禁用；新架构0个可执行任务、最多2个未批准占位。任何扩展需用户与资源所有者另行同意。禁止外部训练数据、隐藏测试标签、Phase15 Holdout调参、自动提交排行榜。

## 唯一正式启动入口（获批准后由朋友执行）

先按README_friend设置数据、协议附件、ImageNet权重、输出/备份路径和GPU预算：

```bash
bash scripts/friend_start.sh --config configs/batch_plan.yaml
```

该入口自动安装检查、导入冻结协议附件、验证数据/权重哈希、GPU smoke与测速、有限队列、独立验证和OOF、报告、分卷打包。缺少附件、版本错误或预检失败会停止；不会重新划分患者。

## 准备与审计命令（不会训练）

```bash
python scripts/protocol_assets.py --config configs/batch_plan.yaml --action import --file /authorized/phase17_protocol_assets_v1.tar.gz
python scripts/batch_run.py --config configs/batch_plan.yaml --action validate
python scripts/batch_protocol_audit.py --config configs/batch_plan.yaml --output outputs/protocol_audit.json
python scripts/audit_git_delivery.py --config configs/batch_plan.yaml --output outputs/git_delivery_audit.json
python scripts/batch_run.py --config configs/batch_plan.yaml --action status --root /authorized/outputs/phase17_batch
python scripts/batch_run.py --config configs/batch_plan.yaml --action stop --root /authorized/outputs/phase17_batch
```

`stop`排空当前任务，不立即杀死训练。OOM不改配置，临时I/O最多重试一次。完整恢复含优化器、调度器、scaler、RNG、早停与EMA，恢复边界为完整Epoch。

## 无GPU验收

Linux：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 python -m pytest tests -q
```

Windows PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
python -m pytest tests -q
```

明确使用`-1`，不要用空字符串。GPU恢复一致性测试在此会跳过，朋友启动前的真实GPU预检会执行。合成验收覆盖15个微型任务、最佳状态、恢复、Val/OOF、打包、独立解包和无GPU复算，不代表实际模型训练成绩。

## 数据与安全

Git仅保存代码、配置与哈希协议。原始ZIP、患者/标签索引附件、预训练权重、checkpoint、结果和个人凭据均被忽略。通过比赛允许的独立渠道传递附件；协议附件只导入被冻结的文件，不重新划分。

所有代码变更使用朋友自己的分支与Pull Request；不能覆盖用户主训练目录。push前执行敏感信息与大小检查并取得用户批准。仓库URL及远程未预设，不自动push、创建远程仓库、上传Release或配置LFS。

## 开发边界

`scripts/batch_train.py`是隔离的已验证训练入口快照，只扩展声明的EMA和恢复产物一致性。来源SHA记在文件首行。修改后必须重新审核并更新plan代码哈希；禁止运行中改配置。

`scripts/batch_preflight.py`、`batch_benchmark.py`、`batch_latency.py`只在朋友明确启动时使用其授权GPU。统计与交付实现位于`src/cxr/batch/`，结果包包含可复算代码，不依赖原集群的绝对路径。
