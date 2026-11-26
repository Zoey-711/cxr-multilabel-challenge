# E08 弱类预处理及错误审计

## 已核实的执行链

`src/cxr/data/dataset.py` 的确定性验证/测试预处理为：打开 ZIP 图片 → PIL 解码 → RGB 三通道 → `ImageOps.pad` 按比例双线性缩放至 384×384、居中填黑边 → `to_tensor` → ImageNet 均值/标准差归一化。随机增强仅用于训练；此次冻结 E08 五个模型的 manifest 均记录 `black_pad_v1`。未见固定的中心裁剪或非等比例拉伸。来源：`src/cxr/data/dataset.py`、`outputs/submissions/E08_EMA_20261020/manifest.json`。

按元数据宽高计算全部 859 个验证 Study 的各图黑边占比，结果见 `weak_class_error_analysis.csv`。对排序错误较重的样本，脚本从原始 ZIP 只读检查 18 张图片的像素和实际尺寸：18/18 可读，18/18 与官方 `Rows/Columns` 一致；抽样黑边占比中位数约 0.168。图像审阅清单见 `local_image_quality_review.csv`，未保存或上传胸片。**这只排除了抽样范围内的文件/尺寸问题，不构成标签质量或肺野完整性的人工影像诊断。**

三个弱类的阳性与阴性 Study 的黑边占比中位数均约 0.168。仅在阳性 Study 中，用排序错误比例与平均黑边占比作探索性 Pearson 相关，类别 8/4/1 分别约 -0.007/-0.075/+0.245；与原图较短边像素数的相关分别约 -0.187/+0.086/+0.143。类别 1 样本仅 24 个阳性 Study，所有这些相关值未做病例构成校正，**未发现足以确证黑边或缩放造成弱类错误的证据**。来源：`weak_class_error_analysis.csv`（按 `label=1` 分组计算）。

## 弱类与视角

| 类别 | 单图 Study AUC（阳性数） | 多图 Study AUC（阳性数） | AP 视角存在 AUC | PA 视角存在 AUC |
|---|---:|---:|---:|---:|
| 8 Lung Opacity | 0.6593（137） | 0.7157（77） | 0.6777 | 0.7545 |
| 4 Pneumonia | 0.6691（37） | 0.8200（31） | 0.6578 | 0.8796 |
| 1 Enlarged Cardiomediastinum | 0.6016（15） | 0.8388（9） | 0.6287 | 0.8585 |

这些分组具有不同病例构成和阳性率。例如类别 8 单图阳性率 137/391，多图为 77/468；AP/PA/LATERAL 组也可相互重叠。因此差异是描述性信号，**不能证明多视角聚合或黑边是因果原因**。完整支持数、患者数及 AP 见 `study_view_stratification.csv`。类别 1 仅 24 阳性 Study、16 阳性患者，分组结果波动尤其大。

## 标签冲突和错误线索

Pneumonia 的 68 个阳性 Study 中，10 个同时标为 Consolidation，36 个同时标为 Lung Opacity，2 个三者共现；Consolidation 与 Lung Opacity 共现 11 个。No Finding 与疾病阳性共现 0。来源：官方 ZIP 中 `valid.csv`，统计代码 `analyze_phase19.py`，结果 `weak_label_cooccurrence.json`。这些是**标签共现**，尚不能证明标签错误；本轮未对报告文本进行复核。

对每个阳性 Study，`weak_class_error_analysis.csv` 计算其概率低于多少比例的阴性 Study（排序错误比例）。类别 8/4/1 的阳性 Study 平均排序错误比例分别为 0.276/0.249/0.272。审阅优先级可按该列排序，但不得据此修改隐藏 Test 概率。E07/E08 同一 Study 的预测差保留在清单中，便于后续在有权限的本地影像审阅。

## 实现开销线索

`src/cxr/inference/predict.py` 的单模型接口对每个模型重新创建 DataLoader；若部署时逐一调用，可能重复 ZIP 读取、解码和预处理。已有 E08 离线候选复用了冻结 E03 Test 概率，但实际新图像推理不能依赖该缓存，仍须执行五个模型。此次分段测速使用**共享一次预处理**的独立审计脚本，尚未修改原始推理代码；若重构为共享张量，必须重新核对逐 Study 概率及 AUC。来源：`scripts/generate_e08_ema_candidate.py`、`src/cxr/inference/predict.py`、`benchmark_e08_segments.py`。
