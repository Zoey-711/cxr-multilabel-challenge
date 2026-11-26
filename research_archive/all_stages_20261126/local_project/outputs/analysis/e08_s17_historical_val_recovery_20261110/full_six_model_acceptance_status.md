# 六模型完整一致性验收状态

截至 2026-11-10：`BLOCKED_TARGET_T4_TENSORRT`，不得宣称历史 78.81 提交的一致性已通过。

## 训练状态

本地 Mixup 队列 `outputs/experiments/mixup_confirmation_20261108/queue_status.json` 为 `DONE`；BASE、MIXUP 均为 `TRAIN_DONE`。未发现正在运行的 Python 训练进程。用户要求停止训练时，任务已自然完成，因此没有杀进程或改写训练结果。

## 已完成的本地核验

- 历史提交 `outputs/submissions/E08_075_S17_F3_025_20261110/submission.csv` 的 SHA-256 为 `8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c`，包含 1414 个 Study、14140 行。
- 官方 Test 原始 ZIP 中的 2218 张图片逐张与冻结的图片清单核对：ID、Study 映射、文件字节数及 SHA-256 全部匹配；图片原始字节共 3,676,899,103。
- 六模型 T4 包的 36 个清单文件、归档成员与 Python 语法通过核验。完整复算脚本 `full_test_audit.py`、测速 `cost.py` 和根目录 `predict.py` 引用同一个六模型加载及单图推理函数。
- 在 4060 上此前完成过全 2218 张 Test 的纯 PyTorch FP16 batch=1 复算，结果见 `outputs/analysis/e08_s17_consistency_standardization_20261110/test_audit/audit_report.json`：1414 Study 中有 10 个类别排序变化，0 个二值判定变化，最大概率差 0.00091552734375。该实现未通过规则要求，不能当作最终部署入口。
- Windows 环境没有 TensorRT，原 Phase65 的 B02/M02 T4 FP32 引擎不能在这里运行；无法用本机得到六模型混合路径的真实 Test 概率。没有执行 GPU 训练或重训。

## 新版目标机验收包

`outputs/deployment/e08_s17_t4_batch1_full_audit_v2_20261110.tar.gz`：362,633,686 字节，SHA-256 `6958559ff02305946ee9e2af0cf6fe70600800a1c5b425e782ab39983aedaaa8`。

在目标 T4 提供原始 Test 的 2218 张 JPG，运行：

```bash
python full_test_audit.py --images-root /private/test_images --output reports/full_test
```

脚本先按内置清单核对全部原始图片，再逐张执行正式六模型推理，Study Mean 后与历史提交的 14140 个概率逐项比较，报告概率误差、类别排序、0.5 判定和固定 10 张图片的重复性。`report.json` 为 `PASS` 才能继续官方 100 张计时核验。若 T4 引擎不兼容或有 1 个 Study 排序变化，必须如实标记失败；不能通过改融合比例或改历史 CSV 来制造通过。

`cost.py` 的官方 100 张核验仍需另外运行。当前包大小核算为 392,800,139 字节，低于 500,000,000 字节；实际 T4 耗时尚未得到。
