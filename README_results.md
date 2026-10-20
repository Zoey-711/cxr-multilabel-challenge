# Phase 17 结果文件夹与本地导入

朋友交付的是完整的 `phase17_batch/` **文件夹**，不生成压缩包。请通过双方授权的大文件传输渠道取得，不放入普通 Git 仓库；不需要朋友逐个寻找模型。

## 目录

```text
phase17_batch/
  manifests/       固定划分、标签、环境、计划、delivery_manifest.json
  experiments/     每模型最佳纯 state_dict、resume、日志、Val/OOF/Test 概率及 SHA
  oof/             完整五折 OOF 与 Fold 来源索引（仅五折齐全时）
  validation/      官方 Val 五折与固定 E17 概率
  test_predictions/  各骨干五折及固定 E17 无标签 Test 概率
  analysis/        AUC/AP、患者级 Bootstrap、互补性与分组报告
  deployment/      真实权重大小、研究/官方目标时延证据及资格表
  submissions/     四列、全十类候选 CSV、独立 manifest 和格式校验表
  reports/         22 项可执行任务状态、总报告及简短完成通知
```

`experiments/<ID>/best_model.pth` 是推理权重；`checkpoint_resume.pth` 含优化器、调度器、AMP scaler、RNG 和早停状态，**仅用于恢复训练**。EMA 文件与原始权重明确分开。Val/OOF/Test 的 CSV 均为一 Study 一行，列是 `Subject_id,Study_id,Prob_1..Prob_10`。Test 没有标签，也没有 Test AUC。

G/H 四个全 Train 任务属于默认26项计划，需朋友明确填写总GPU小时和G/H子预算；依赖与4090短步预检通过后由同一启动命令执行。结果保存在各自的 `experiments/<ID>/`，状态另见 `extension_state.json`、`reports/extension_status.csv`、`reports/gh_experiment_summary.csv` 和 `analysis/additional_experiments_summary.csv`。G/H 的 OOF 指标始终为 `N/A`，因为全 Train 模型没有未见患者折外预测。完整五折 OOF 只来自 D/C/S/N 的各五折模型；`analysis/gh_teacher_oof.csv` 是患者隔离的训练软标签，不是学生OOF。若依赖、预检或预算失败，相关任务为 `BLOCKED`，不得填入虚构指标。

## Windows 本地验证与导入

先在本机从已审核代码分支安装项目（不需要 GPU 即可检查/复算统计），再将朋友交付的文件夹放在不覆盖原本地训练的位置。PowerShell 示例：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/import_friend_results.py --config configs/batch_plan.yaml --source 'D:\received\phase17_batch' --version friend_v1 --expected-commit '<朋友运行的固定Git commit>'
python scripts/batch_delivery.py --config configs/batch_plan.yaml --action verify --root .\outputs\phase17_friend_import\friend_v1
python scripts/batch_delivery.py --config configs/batch_plan.yaml --action recompute --root .\outputs\phase17_friend_import\friend_v1
```

导入脚本先核对代码 commit、数据/折划分版本、实际配置及模型冻结 SHA，再复制到 `outputs/phase17_friend_import/friend_v1/` 并复核；目录已存在时拒绝覆盖。纯 CPU 的 `recompute` 按 Subject_id/Study_id 关联冻结标签，重新计算 Val/OOF、固定融合和 Bootstrap。原图不在结果目录里；统计分析不依赖朋友的绝对路径。若要对测试图像重新推理，仍需合法原始比赛数据和对应权重，并遵守比赛规则。

## Submission 与部署

候选 CSV 是四列 `Subject_id,Study_id,Predict_class,Probability`，每个 Test Study 对应类别 1–10 共十行，不筛阈值、不加 Gate。脚本 `scripts/validate_submission.py` 通过格式校验才标 `FORMAT_PASS`；它**不会上传比赛平台**。`submissions/submission_candidates.csv` 同时列出 Val Macro-AUC、模型权重字节、完整单图时延证据以及 `SIZE_FAIL`、`LATENCY_FAIL`、`LATENCY_UNVERIFIED`、`ACCURACY_PENDING`、`PASS` 状态。朋友 4090 上的速度是研究参考，缺官方目标设备及完整口径实测时不可称为合规。已知超 500MB 或超过 100ms 的候选只能留作研究；80ms 是内部安全目标。

任何任务缺失、失败或 Test 未完成都必须在状态表与 `manifests/delivery_manifest.json` 中保留原状；不填补不存在的指标。官方 Val 曾多次用于模型选择，Bootstrap 差值属于探索性证据，不是新的独立测试结果。Phase 15 内部 Holdout 不再用于本批任务选择模型。
