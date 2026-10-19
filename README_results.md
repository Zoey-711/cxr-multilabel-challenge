# Phase17 结果包（中文）

结果与模型分开交付。`delivery_manifest.json` 是唯一包清单；先校验所有分卷，不凭文件名选择权重。`COMPLETE.txt` 会列出未完成任务。没有收到并校验完关键产物前，不删除远端结果。

## 文件用途

- `results_bundle.tar.gz`：配置、标签索引（不含原图）、Val/OOF概率、统计、日志、代码与报告。
- `models_bundle.tar.gz`（或 `.part0001` 等分卷）：纯 `state_dict` 推理权重及配置。
- `resume_bundle.tar.gz`（可选）：`checkpoint_resume.pth`，包含优化器、调度器、scaler、RNG、早停等。不能当作推理权重。
- `phase17_batch/experiments/D17_F01/best_model.pth`：普通最佳模型；`ema_best_model.pth` 属于独立EMA候选。
- `phase17_batch/manifests/train_labels.csv`、`val_labels.csv`：固定标签与ID，用于离线复算；不按行序配对。

## Windows 下载、校验和解包

通过双方批准的网盘、对象存储或大文件传输渠道，接收整个delivery版本目录，包含manifest及全部分卷。使用该渠道提供的断点续传功能；脚本负责下载后的完整性校验，不登录朋友设备。不要把原图或大型模型提交Git仓库。不自动启用GitHub Release/LFS。

```powershell
Get-FileHash -Algorithm SHA256 .\results_bundle.tar.gz
# 与delivery_manifest.json对应值比较。下面工具还会校验全部包和解包后的每个文件。
python scripts/import_friend_results.py --config configs/batch_plan.yaml --bundle D:\received_delivery --version friend_v1 --expected-commit <双方冻结的代码commit>
```

默认导入 `outputs/phase17_friend_import/friend_v1/`，目标必须尚不存在。脚本先核对代码commit、数据与划分版本、模型ID，再按分卷顺序拼接，验证SHA-256，阻止路径穿越、符号链接和不同内容覆盖。空间需容纳解包后的权重及临时拼接包。下文的 `D:\verified_phase17` 请替换为实际导入目录。

## 不用GPU复算

使用本地现有Python环境（需要numpy、PyYAML）。结果包自带评价代码，源集群绝对路径不参与计算：

```powershell
python -m pip install --no-deps -e D:\verified_phase17\code
python D:\verified_phase17\code\scripts\batch_delivery.py --config D:\verified_phase17\phase17_batch\manifests\batch_plan.yaml --action recompute --root D:\verified_phase17\phase17_batch
```

这会重算Study指标、完整OOF、固定融合及患者bootstrap（2000次有效抽样），不读取胸片、不调用GPU、不重新训练。报告目录可重新计算；冻结实验文件不能修改。仅有结果包也能复算，模型大小读取经过校验的manifest。

## 加载最佳推理权重

在安装项目和PyTorch后，使用解包配置中的模型名。权重本身没有原始数据路径或pickle模型对象：

```python
from pathlib import Path
import torch, yaml
from cxr.models.factory import create_model
run = Path(r'D:\verified_phase17\phase17_batch\experiments\C17_F01')
config = yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))
model = create_model(config['model']['name'], pretrained=False)
model.load_state_dict(torch.load(run/'best_model.pth', map_location='cpu', weights_only=True), strict=True)
model = model.cuda().eval()
```

未来测试集推理使用 `.pth` 纯推理权重、对应预处理配置和固定融合公式。必须另行获得用户批准；本批次没有生成排行榜submission。

## 选择500MB候选

查看 `ensembles/deployment_candidates.csv/json`，用真实推理文件大小判定；完整15模型只是研究方案。依赖清单包含路径与SHA-256。保留小量预定义候选，不做权重搜索。普通与EMA分开计算，不混用。

官方Val已反复用于Epoch与模型选择，其差值区间不是新的独立确认；OOF是每患者对应留出Fold的预测，不是该患者全部五折模型的平均。T4端到端延迟仍需在目标设备上测量。
