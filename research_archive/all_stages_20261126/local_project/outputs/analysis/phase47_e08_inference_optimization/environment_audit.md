# Phase47 环境与规则核查

- 本机 Windows、RTX 4060 Laptop 8GB，驱动 581.08；PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128、CUDA 12.8、Pillow 12.3.0、NumPy 2.5.2。隔离目录的 ONNX Runtime GPU 1.23.2；原图 opset 17，输入 1×3×384×384。
- 当前 ORT SessionOptions 默认 `ORT_ENABLE_ALL`。CUDA Provider 位于 session 第一位，但这不足以证明每个节点都在 GPU 上；逐节点 profiling PENDING。
- Python TensorRT 与 `trtexec` 均不存在。虽然 ORT 构建列出 TensorRT EP，本机无法据此认定 TensorRT 引擎可用；未安装、未构建、未测速。
- 跑完基准后发现另有 `F:\glaucoma-bdc-2026\.venv\Scripts\python.exe` 进程使用 GPU；随后 4060 利用率约 95%、显存约 5.3GB。Phase47 后段分段和配对计时受资源竞争影响，不可与早段孤立基准作严格绝对值比较。没有停止该进程，也未继续做新 GPU 实验。
- 分段墙钟计时见 `latency_breakdown.csv`：受竞争影响的一轮中，三次 DenseNet 前向各约 38.7–38.9ms，解码约 25.35ms、预处理约 32.76ms。ORT `session.run` 内部的 H2D/D2H 与计算尚未分离；独立 GPU kernel 时间未测，不得把这些前向墙钟数当作纯 GPU 计时或未受干扰的耗时占比。
- [官方规则 PDF](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)第6–8条允许固定参数前向与固定多模型集成，要求 T4 指定环境的 5 张预热、100 张固定图、batch=1、连续3轮，读图至最终结果写出；实际调用模型与必要配置解压后 ≤500MB。本地 Val 图片不是官方固定 Test 计时图。
- 预训练权重上游许可证全文及原始文件 SHA 的最终材料复核仍未完成，详见 `E:\计挑赛\cxr-multilabel-challenge\outputs\analysis\phase29_deployment\pretrained_weights_compliance.md`。
