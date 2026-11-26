# 固定 Mixup：Fold2 成对确认

仅运行 BASE → MIXUP 两项，单个 GPU 顺序执行。实验方案见 `outputs/analysis/mixup_confirmation_proposal_20261108/plan.md`。

从项目根目录执行（Windows）：

```powershell
.\.venv\Scripts\python.exe scripts/run_mixup_confirmation.py --config configs/mixup_confirmation_20261108.json --action audit
.\.venv\Scripts\python.exe scripts/run_mixup_confirmation.py --config configs/mixup_confirmation_20261108.json --action smoke
.\.venv\Scripts\python.exe scripts/run_mixup_confirmation.py --config configs/mixup_confirmation_20261108.json --action queue
```

已有队列运行时禁止再次执行。`queue.lock` 记录进程 PID；进程仍存在时不可删除锁。异常终止后需先核实 PID、失败日志和检查点完整性再恢复。

## 冻结协议

- EfficientNetV2-S ImageNet 初始化，384，micro-batch 8，累积 2；BN 按微批更新，两组一致。
- AdamW backbone 1e-4 / head 3e-4，weight decay 1e-4；warmup 2，cosine 30；EMA .999，包含 buffers。
- FP16 GradScaler 初始 scale 1024，先 unscale 再检查及裁剪；仅成功更新后推进 scheduler 和 EMA。连续 3 次或累计超过 20 次 AMP 跳步即失败，不静默掩盖不稳定。
- Mixup 固定 alpha .2，每个微批使用单一 Beta 系数和独立随机流。无模型/融合搜索。
- 单进程数据读取，两组相同；独立 Mixup RNG 不改变数据增强、模型随机层及数据顺序的 RNG。
- 剔除原 Phase15 锁定 Holdout：其历史索引中的 `internal_val`，来源索引 SHA-256 已核验。
- 外层 Fold2 完全不参与训练和早停；其余患者按预先声明的 SHA-256 排序取前 10% 作为内部早停集。
- 最多 30 epoch，EMA 内层 Study AUC 早停 patience 6；不访问官方 Val/Test。

## 恢复及结果

每轮原子写入 `last_training.pt`：模型、EMA、优化器、调度器、GradScaler、早停、历史、Python/NumPy/Torch/CUDA/DataLoader/Mixup RNG。只能从完整 epoch 边界恢复；中断 epoch 会重放。

两项均完成后才评估外层 Fold2，生成 `outer_oof_predictions.csv`、各类别指标、2,000 次患者配对 Bootstrap 与 `final_decision.md`。Bootstrap 中缺乏任一类别正负样本的重采样记为无效并报告，不更换抽样以隐藏问题。

初筛条件：Macro-AUC 差值 ≥ .001，任何类别下降不超过 .01。通过仅记为 PROMISING_PILOT，后续 Fold3–5 不自动启动。该 Fold 曾用于历史研究，结果不等同全新盲测；也不是完整五折 OOF。

实时进度：`BASE/status.json`、`MIXUP/status.json`、各自 `train.log` 和队列标准输出。原 E08、H17、历史实验不改动，不推送 GitHub，不生成 Submission。
