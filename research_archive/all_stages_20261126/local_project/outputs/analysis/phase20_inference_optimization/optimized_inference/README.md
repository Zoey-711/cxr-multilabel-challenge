# 已核验的独立推理实现

`shared_preprocess.py validate` 调用同目录上级 `run_phase20.py` 中的共享输入实现。它读取冻结的五个 best checkpoint，逐个验证 SHA-256，**每张图只解码、Resize/Pad、Normalize 一次**，依次给五模型推理；每个模型维持其原独立验证的 batch 大小、FP16、逐图 Sigmoid、Study Mean 与冻结的嵌套融合次序。原正式推理路径、checkpoint、Submission 均未修改。

在 1,422 张官方 Val 图像、859 个 Study 上，五个基础模型及 E03/E08 的全部 Study 概率与冻结文件逐项完全一致（最大绝对差 0），逐类和 Macro-AUC 也完全一致。证据：`../prediction_equivalence.json`。固定 8 张不同尺寸和视角图像的 batch=1 配对测速见 `../baseline_benchmark.json`。

这是**本地验证过的独立工程原型**，不是官方 T4 合规证明。`channels_last` 研究分支未通过预设全量一致性容差，未并入该实现。真实 Test/复赛部署接口和目标硬件时延仍须分别验收。
