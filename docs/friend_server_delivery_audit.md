# Phase32 + E08-Tune交付核验

2026-10-24，本地基线 `da83d215e4667981da94c1423eb005443be8c959` 为 `friend-gpu-batch`，其父系包含Phase32提交 `bd3741e196b341f1eeb8ddde201af194e3508955`。发现两套队列各自只有输出目录锁，在同机同时启动时仍有GPU抢占窗口。此次仅增加两者共用的主机GPU锁、独立真实预检动作及交付说明；未更改任何训练超参数、五折配置、模型或原有输出。

| 检查项 | 结果 |
|---|---|
| Phase32 | 8项配置、`scripts/phase32_run.py`、README、Epoch断点与Val/OOF导出齐全 |
| E08-Tune | 5项Fold1 Pilot、至多2项条件补齐的模板、`scripts/e08_tune_run.py`、README、断点与EMA Val/OOF导出齐全 |
| 任务ID | P32_*与ET_*无交集；历史实验ID不复用 |
| 输出/日志/内部锁 | 默认独立phase32_batch与e08_tuning；各自queue.lock、状态、日志、恢复checkpoint |
| GPU | 每队列单GPU绑定；新增共享Linux `/tmp/cxr_friend_gpu_queue.lock`，第二队列同时启动直接拒绝 |
| CPU/RAM/磁盘 | 每队列开跑前核查逻辑CPU、可用内存及输出盘剩余量；两队列依次运行 |
| GPU小时 | 两个独立显式预算；E08另有补折子预算，默认null均阻止启动；朋友需批准合计上限 |
| 结果同步 | 默认关闭。授权时分别上传到results/phase32-pilots与results/e08-tune，独立checkout，不上传影像或权重 |

交付CPU回归命令：

```text
python -m pytest tests/test_friend_host_queue_lock.py tests/test_phase32.py tests/test_e08_tune.py tests/test_early_stopping.py tests/test_metrics.py tests/test_patient_folds.py -q
```

结果：**68 passed，0 failed，0 skipped**。CUDA_VISIBLE_DEVICES=-1；跨进程持锁/释放、真实预检命令不会进入正式队列、计划SHA、预算、损失与EMA/OOF合成验收已通过。未在本机运行Linux4090训练或正式前反向；朋友服务器仍须通过两批`--action preflight`。本地CPU测试不构成T4延迟合规证据。

注意：共享锁只覆盖Phase32和E08-Tune新入口，旧Phase17队列或其他GPU作业须由朋友依其调度规则错开。两个本地配置必须指定不重叠的输出与上传checkout目录。GPU小时预算各自生效，不能把两个独立上限误当成共同总上限。
