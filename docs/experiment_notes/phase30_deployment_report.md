# Phase30 部署优化与 T4 验收状态

- 已从旧推理的双 `convert('RGB')` 删去外层一次；1422 张 Val 输入张量完全一致，固定 100 图的成对前向概率完全一致。未改模型、384 分辨率、融合权重或 Study Mean。
- 优化前后 4060 `cost.py` 测试使用相同 100 张 Val 图片、batch1、5 预热、3 轮；两次运行不是严格配对，归因以独立分段配对数据为准。

- E08_EMA：原 `cost.py` 182.39ms，优化版 164.83ms（P50 164.88 / P95 179.96 / max 186.98ms）；分段交替配对 177.15→167.18ms。
- H17_02：原 `cost.py` 100.38ms，优化版 88.46ms（P50 89.24 / P95 98.79 / max 105.74ms）；分段交替配对 108.01→96.14ms。

重复独立运行存在波动：E08 优化版均值 164.83–166.61ms，H17 优化版均值 88.46–93.77ms；交替配对结果更适合归因。记录见 `timing_repeatability.json`。

模型加必要配置及源代码的保守字节数见 `model_size_and_hashes.json`，均低于 500,000,000 字节。

本地只有 RTX4060；无 T4、无官方固定 100 张计时图片及对应完整 Test Study 映射。因此两者均为 **T4_UNVERIFIED**，对原冻结 Submission 的本地 100 单图 Study 一致性也未通过。对已有冻结 Val 概率的 1e-4 阈值均为 FAIL；不得标记正式部署 PASS。

候选代码位于 `candidate_deployment/`；`cost.py` 确实导入同目录 `predict.py` 的加载与单图推理函数。其结果 CSV 包含完整文件读取到写出概率的墙钟时间。见 `candidate_deployment/README_T4.md` 复现命令。
