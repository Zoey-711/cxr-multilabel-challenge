# 六模型 T4 候选包：本地预检

状态：`LOCAL_PARTIAL_PASS`；完整六模型推理和历史 78.81 提交一致性 `UNVERIFIED`。

使用项目 `.venv`（Python 3.12.4，PyTorch 2.10.0+cu128）执行。4060 当前显存占用约 2438 MiB；进程明细权限不足。本轮仅进行 CPU 前向，不占用训练 GPU。

| 核验 | 实际结果 |
|---|---|
| 压缩包 | 362,499,071 字节，33 个成员；可读取，路径安全 |
| SHA-256 | 包内 32 个文件全部匹配 manifest；历史提交 SHA 为 `8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c` |
| 参考提交 | 1414 Study、14140 行，通过结构读取 |
| 样本 | 100 张计时 Val 图和 5 张独立预热 Val 图全部真实解码；索引 100 行与文件名一致 |
| CPU 模型前向 | B01、B03、N01_EMA、S17_F3 权重均成功加载；同一真实 Val 图预处理后输出 10 个有限且位于 [0,1] 的概率 |
| 推理入口 | 根目录 `predict.py` 与 `cost.py` 导入同一 `load_models` 和 `predict_image` 函数对象 |
| 完整 `cost.py` | 本机直接执行，退出码 1：`ModuleNotFoundError: No module named 'tensorrt'`。`check_environment.py` 同样在此处退出 |

本机没有 TensorRT，且包内 B02/M02 为 T4 构建的 FP32 引擎。因此本地未运行这两个模型，也未得到六模型预测、历史提交的排序/判定/概率比较或六模型端到端耗时。CPU 模型耗时不代表 T4 性能。需在目标 T4 上运行包内 README 命令，并使用与历史提交对应的 Test 图片及完整 Study 映射完成一致性核验。
