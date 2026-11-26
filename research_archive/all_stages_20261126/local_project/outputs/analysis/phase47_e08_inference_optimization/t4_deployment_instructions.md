# T4 验收步骤（未执行）

1. 在独立 Linux/Python 3.12 环境安装 `requirements_inference.txt`，并固定与官方指定 CUDA/cuDNN/驱动兼容的版本。复制 Phase45 `predict.py`、`cost.py`，五个 FP32 ONNX 图、导出哈希清单及 `src/cxr/data` 预处理代码；按原项目相对目录布置。切勿复制本机 Windows `isolated_deps` 作为 Linux 依赖。
2. 按 SHA-256 验证所有图、代码、配置；确认五个 session 均使用 CUDAExecutionProvider，并运行单图及 859 Study Val 一致性检查。
3. 使用官方 5 张预热与固定 100 张 Test 计时图，运行 `python cost.py submission.csv pictures_100 pictures_5 --test-archive /path/to/cxr_first.zip`。核对实际 T4 端到端均值 ≤100ms、100 张对应预测与提交 CSV 同一入口，并明确精度 FP32。
4. 如尝试 TensorRT，必须在 T4 兼容环境针对 1×3×384×384 逐图构建五个引擎；重新跑完整 Val 数值/AUC 与 Test 预测并生成该引擎独立 Submission。4060 引擎不能视为 T4 合规证明。
5. 当前 `cost.py` 对不完整 Study 样本只标记等价性未验证，不能把部分图像概率直接与 Study Mean Submission 比较。
