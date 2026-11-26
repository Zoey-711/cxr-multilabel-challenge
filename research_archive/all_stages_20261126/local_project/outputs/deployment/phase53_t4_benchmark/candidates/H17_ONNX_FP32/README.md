# H17 ONNX FP32

独立 H17_DISTILL_02 ONNX CUDA FP32 引擎，384黑边填充，Study Mean。模型为 `h17_raw_fp32.onnx`；`predict.py` 包装 Phase49 已验证入口，`cost.py` 调用同一入口。该精度版本没有已确认的对应官方 Submission，默认 `SUBMISSION_CONSISTENCY_UNVERIFIED`；若以后生成匹配版本，可用 `--submission` 指定。T4_UNVERIFIED。
