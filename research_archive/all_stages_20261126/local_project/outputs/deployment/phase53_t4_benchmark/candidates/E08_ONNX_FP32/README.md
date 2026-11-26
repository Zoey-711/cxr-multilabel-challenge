# E08 ONNX FP32

独立5图 E08 固定融合；输入384黑边填充，ImageNet归一化，ONNX CUDA FP32，Study Mean。模型为 `onnx_models/` 中五个 FP32 图；运行时不加载原始 PyTorch checkpoint。`predict.py` 包装已冻结的 Phase45/Phase44 推理实现，`cost.py` 调用同一入口。`reference_submission.csv` 与 ONNX FP32 版本对应。全部文件由 `manifest_sha256.json` 核验。T4_UNVERIFIED。
