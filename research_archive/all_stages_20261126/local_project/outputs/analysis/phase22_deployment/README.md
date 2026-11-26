# Phase 22 独立部署准备

此目录复用 Phase20 已在 1,422 张官方 Val 图像、859 个 Study 上逐项验证的共享预处理实现。冻结 E08/E03 checkpoint 只通过原实验路径读取并逐个核对 SHA-256；不复制或修改权重、原始图片、原 Submission。推理顺序与冻结公式保持一致：逐图 sigmoid、Study Mean，再按 E03/E08 权重融合。

本目录的 `benchmark_t4.py` 测量单张图像从官方 ZIP 读取、RGB 解码、384×384 等比例缩放与黑色 Padding、ImageNet 归一化、数据传输、所有参与骨干前向、融合到单图 Study Mean 的完整流程。模型在计时前加载，batch=1，预热 3 轮，测量 10 轮；保存均值、P50、P95、最大值及各阶段均值。官方尚未给出模型加载、缓存、预热和统计量的精确计时口径，因此即便在 T4 上运行，此脚本的研究性结果也不能独自证明官方合规。

在项目根目录验证冻结权重与 Phase20 证据（CPU，无 GPU）：

```powershell
.\.venv\Scripts\python.exe outputs\analysis\phase22_deployment\benchmark_t4.py --config outputs\analysis\phase22_deployment\deployment_config.json --candidate E08-EMA --mode verify
.\.venv\Scripts\python.exe outputs\analysis\phase22_deployment\benchmark_t4.py --config outputs\analysis\phase22_deployment\deployment_config.json --candidate E03 --mode verify
```

在装有相同项目、比赛 ZIP、合法 checkpoint 和 CUDA PyTorch 的 T4 Linux 环境中运行测速：

```bash
python outputs/analysis/phase22_deployment/benchmark_t4.py --config outputs/analysis/phase22_deployment/deployment_config.json --candidate E08-EMA --mode benchmark --archive /ABS/PATH/cxr_first.zip --output outputs/analysis/phase22_deployment/e08_t4_benchmark.json
python outputs/analysis/phase22_deployment/benchmark_t4.py --config outputs/analysis/phase22_deployment/deployment_config.json --candidate E03 --mode benchmark --archive /ABS/PATH/cxr_first.zip --output outputs/analysis/phase22_deployment/e03_t4_benchmark.json
```

运行前先确认 T4 16GB 能同时装载全部参与模型。脚本拒绝覆盖已有测速结果。正式复赛推理部署仍须取得官方完整计时规范，并在目标容器实测；在此之前部署状态为 `LATENCY_UNVERIFIED`。没有引入 ONNX、TensorRT 或量化。

已验证概率一致性的证据：`../phase20_inference_optimization/prediction_equivalence.json`。本地 4060 共享预处理均值为 E03 158.69ms、E08 179.67ms；这不是 T4 合规结论。
