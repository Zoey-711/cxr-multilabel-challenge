# Phase32本地交付验收（2026-10-24）

独立开发目录：cxr-phase32-friend；分支friend-gpu-batch。基于远程2d90ef64d0a39ce188d9aebb7dcb0327dc91f2dc，不改main、历史结果分支或主项目冻结实验。

最终CPU回归：**27 passed，0 failed，0 skipped**。命令：

```text
python -m pytest tests/test_phase32.py tests/test_early_stopping.py tests/test_metrics.py tests/test_patient_folds.py -q
```

CUDA_VISIBLE_DEVICES=-1，OMP_NUM_THREADS=1，MKL_NUM_THREADS=1。最终集合明确不包含CUDA测试。测试涵盖8配置与唯一变量、所有损失有限梯度、Mixup与模型/AdamW/调度器/RNG完整CPU恢复、训练Study权重计数、384/512像素输入不变及448形状、预算拦截、独立GPU槽Mock调度、有限重试、OOM失败保留、DONE不重复执行、859/3687合成概率、模型导出、异目录重载和SHA篡改检测、轻量导出排除权重影像。

Windows沙箱首次pytest临时目录遇到WinError5；获准在独立目录运行后测试通过。入口--help及git diff --check已执行。未安装或升级本地依赖，未启动正式模型训练。

**仍待朋友实机验证**：Linux导入、8张4090实际绑定、真实完整数据读取、8项单步前反向、AMP、CUDA checkpoint恢复、真实Val/Test短推理、实际显存、并发IO吞吐、长期训练恢复和GitHub结果上传。代码把真实单步验证作为正式训练门禁；本地合成通过不能冒充4090通过。

研究延迟脚本是3次预热/20次计时的单模型参考，非官方成本验收。T4完整读图到写出≤100ms仍未验证。全部部署权重与必要配置必须≤500000000字节。

发布只包含源码、配置、说明与历史聚合指标/哈希；不含原始标签、图像、checkpoint、Submission或本机授权配置。不会自动提交比赛。增量结果上传默认关闭，须朋友明确配置授权，且仅允许独立私有results/phase32-*分支。
