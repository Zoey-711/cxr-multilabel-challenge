# Phase65代码借鉴范围

读取来源：`F:/PHASE65_RANDOM_BENCHMARK_20261102/PHASE65_FULL_REPRODUCTION_PORTABLE_20261102.zip` 中 `PHASE65_FULL_REPRODUCTION_PORTABLE_20261102/phase65_source/predict.py`。

该文件SHA-256：`fcc65dc88ee54b9ce22c166435d71b117c48c1ef68dd57d206ca9fcbefdba681`，与此前审计保存的源码逐字节一致。

原包采用：灰度图先pad/resize再转RGB、静态GPU输入缓冲区、PyTorch FP16 CUDA Graph、一次性返回多模型概率，以及B02/M02的TensorRT FP32引擎。其原E08公式不含本地S17-F3，原T4成绩不能直接赋予新增的六模型组合。

本轮保持B01/B02/B03/M02/N01_EMA/S17_F3全部六个checkpoint，最终公式为 `0.75 E08 + 0.25 S17_F3`。把原包的灰度预处理、集中GPU到CPU传输、CUDA Graph依次实现为独立候选；不更改384输入、归一化、类别顺序或融合权重。所有模型均使用原FP32存储权重和FP16 autocast，未使用TensorRT替换算子。

当前Windows环境PyTorch=2.10.0+cu128，CUDA运行时12.8，GPU为RTX4060 Laptop 8GB。本机主环境未安装TensorRT，未安装新依赖。包内TensorRT引擎对应T4/Linux，不能直接作为本机4060的有效引擎。因此混合TensorRT方案本轮标记BLOCKED_NOT_INSTALLED_AND_TARGET_ENGINE_MISMATCH；CUDA Graph在当前GPU加载时即时捕获，不保存或复用T4二进制引擎。

基准与全部候选使用同一个Phase53计时核心：5张独立预热图、100张原始Val图、batch=1、连续3轮，每张三轮平均后再整体平均。读图、解码、预处理、H2D、全部模型前向、固定融合、单图预测CSV写出/flush及CUDA同步计时；模型加载和graph捕获属于初始化，不计时。保存每次300条计时和概率；Study聚合在全Val一致性审计中检查，不在单图benchmark中缓存其他图片概率。

计时图来自预先固定的Val数据包，不使用Phase65包内Test图片挑选优化方案。最终选择候选必须完成1422张Val图输入及859个Study概率检查，并复测未修改的基准和候选。容差在运行前固定为1e-4；若完全一致则报告EXACT_MATCH，否则如实报告误差。相对78.81提交时的历史batch/cached概率另行报告，不把batch=1内部等价误称为与历史Submission等价。
