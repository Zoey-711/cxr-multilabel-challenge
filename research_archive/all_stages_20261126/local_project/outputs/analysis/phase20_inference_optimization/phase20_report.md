# Phase 20：冻结 E08 推理优化核验

## 官方约束与测量边界

[官方人工智能挑战赛赛题](https://www.ncccu.org.cn/index/Paper/case1.html?way=sichuan)明确列出 T4 16GB、单张图像端到端 ≤100ms 和压缩前模型总大小 ≤500MB；页面未给出预热轮数、计时起止范围、batch/精度或使用均值/P95/最大值的细则。本轮只在 RTX 4060 Laptop 上做工程配对测量；不能宣称官方目标硬件合规。计时从官方 ZIP 读取单张图片开始，到五/四模型概率融合及单图 Study Mean 结束，模型预先加载；单图 Study Mean 不是多图 Study 的整检查耗时。CUDA 在完整计时前后及模型前向后同步。来源：`baseline_benchmark.json`、`latency_breakdown.csv`。

固定 8 张不同尺寸和 AP/PA/LATERAL 视角的 Val 图像，预热一轮、测量两轮（每候选每模式 16 次）；原式顺序重复读图/变换每个模型，共享式每图只读图/变换一次。模式次序交替。两种方案用相同权重、FP16、batch=1、图像、设备和计时实现。它模拟原正式接口每个模型独立调用 `predict_archive_images` 的重复输入过程，**不包含模型加载**；官方是否将模型加载计入 100ms 尚不明确。

| 候选 | 重复输入均值 / P95 | 共享输入均值 / P50 / P95 / max | 实测均值节省 | Val AUC | 权重 |
|---|---:|---:|---:|---:|---:|
| E03 | 365.35 / 430.57ms | 158.69 / 161.36 / 187.83 / 188.90ms | 206.66ms | 0.8099309507 | 196,769,791 bytes |
| E08-EMA | 457.70 / 530.44ms | 179.67 / 187.67 / 213.80 / 221.26ms | 278.03ms | 0.8116686461 | 278,436,014 bytes |

这批图片上的最大模型/融合概率差为 0，逐像素模型输入差为 0。对全部 **1,422 张 Val 图、859 个 Study**，共享预处理使用各模型原独立验证 batch（DenseNet 16、M02/N01-EMA 8），每个模型先逐图 Sigmoid 后 Study Mean，再按冻结嵌套公式融合。五个模型及 E03/E08 的全部 Study 概率、Macro-AUC、逐类 AUC 与冻结文件**完全一致**，最大绝对差 0。来源：`prediction_equivalence.json`。原型只写本目录，未改正式推理代码。

## 其余低风险选项

- **H2D pinned memory**：包含 pin 调用的单次拷贝均值 0.2747ms → 0.2334ms。贡献很小，未作为独立部署变更。来源：`option_probe.json`。
- **inference_mode + FP16 autocast**：原推理已有，不是本轮新增收益。未改变输入 dtype、归一化或推理精度。
- **channels_last**：样本稳态五模型前向约 52.43 → 47.53ms；最大样本概率差 0.0004883。完整 Val 检验 未通过预设容差；E08 AUC 0.8116386350，相对冻结值最大概率差 0.0012614。即便全量数值检查通过，也尚未完成同口径完整端到端配对测速，暂不并入已验证的共享预处理实现。来源：`option_probe.json`、`prediction_equivalence_channels_last.json`（如存在）。
- **ONNX Runtime / TensorRT**：未安装、未转换、未编译；状态 `PENDING_APPROVAL`。ONNX Runtime GPU 需核对 CUDA/cuDNN 匹配的运行包及导出算子支持；TensorRT 还需对应运行库、ONNX 解析和针对目标 GPU 的 engine 构建。五模型的实际转换/编译耗时、峰值显存及 engine 大小**均未测，不能给出可信数字**。默认 TensorRT engine 与构建时 GPU 类型绑定，在 4060/4090 构建的文件不能直接假定可用于 T4；低精度及算子转换也可能改变概率和 AUC。来源：[ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[NVIDIA TensorRT engine 兼容性](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)。应先取得官方计时口径并在 T4/目标环境测量共享式，再决定是否投入额外依赖与构建成本；新后端仍须重新做 859 Study 概率/AUC、模型文件总大小与完整延迟核验。

## 弱类别系统问题只读复核

官方 Val `valid.csv` 的 859 个 Study 没有多图标签字符串冲突。五个 E08 manifest 都为 `black_pad_v1`、384×384；`src/cxr/data/dataset.py` 与此次独立逐像素检查一致。`aggregate_mean_probabilities` 按 Subject/Study 汇聚每张图概率，不使用视角字段改变权重或顺序。类别 8 排序错误最重的 20 个阳性 Study 中 15 个为多图，而全部类别 8 阳性 Study 中 77/214 为多图；这仅是探索性富集信号，可能受病例构成及 Study 标签指向不同视角影响。未找到明确可复现的解析、标签对齐或 Study Mean 实现错误，未修改标签/概率。来源：Phase19 `weak_class_error_analysis.csv`、官方 ZIP `valid.csv`、本轮 `prediction_equivalence.json`。

## 部署判断

E03 比 E08 在本机共享输入路径下均值快 20.98ms，四模型权重也更小；两者均未在官方目标环境按官方完整口径实测，所以均为 **LATENCY_UNVERIFIED**。本机数值高于 100ms 不等于 T4 上必然失败，也不能因 4090 结果低于 100ms 认定合规。目标设备还需对 ZIP/文件读取、预热、批量、精度、计时统计量、单图与多图 Study 分开测，并确认官方是否计入模型加载。除共享输入外，本轮不推荐未验证的进一步优化进入部署。
