# 本地验收记录

2026-10-24，在独立friend-gpu-batch副本开发，未修改主项目冻结结果或既有Phase17/32训练入口。

最终CPU集合：**64 passed、0 failed、0 skipped**，11.70秒pytest运行时间。逐项证据见test_evidence.json。

```text
python -m pytest tests/test_e08_tune.py tests/test_phase32.py tests/test_early_stopping.py tests/test_metrics.py tests/test_patient_folds.py -q
```

环境变量CUDA_VISIBLE_DEVICES=-1、OMP_NUM_THREADS=1、MKL_NUM_THREADS=1。本地没有新正式训练，没有安装/升级依赖。CLI --help和git diff --check已执行。

覆盖25份模板的单变量配置与有限损失梯度、3种EMA衰减的BN缓冲区更新与模型/optimizer/scheduler恢复、最多2项确定性筛选、失败依赖BLOCKED、选择哈希不可改变、GPU预算授权、Mock队列恢复不重跑、完整18513 Study/11055患者合成OOF与Bootstrap、基线篡改拦截、普通/EMA分别导出和异目录重载；已有Phase32协议哈希校验仍通过。

未在本地验证：Linux真实数据读取、朋友GPU绑定与空闲情况、CUDA AMP/完整恢复、训练吞吐/显存、长时间训练、GitHub增量结果上传。必须由朋友运行入口中的REMOTE_PREFLIGHT及实际任务取得证据，不能以CPU测试代替。T4性能始终未验证。

没有原始数据/标签、checkpoint或Submission加入本次代码交付。计划中的25个模板最多执行13项，原始基准复用，不暗中再训练。
