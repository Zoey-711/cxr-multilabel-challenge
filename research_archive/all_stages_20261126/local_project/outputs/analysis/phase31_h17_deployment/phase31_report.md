# Phase31：H17-02 本地部署准备结果

## 完成状态

- 独立候选包：`H17_02_T4_candidate/`，含单个 H17-02 纯推理权重、完整运行配置、复制的 `src/cxr` 代码、共同推理入口 `predict.py` / `cost.py`、`requirements.txt`、README、软件许可证与 SHA-256 清单。已从包目录独立导入并校验入口；不依赖原项目的绝对路径。
- 新候选：`results/test_predictions.csv` 和 `results/submission.csv`。官方无标签 Test 元数据为 2,218 张图、1,414 个 Study；Study Mean 后生成 14,140 行，列名和顺序为 `Subject_id,Study_id,Predict_class,Probability`。独立 ID/概率/类别/宽表核验 PASS；Submission 大小 **647,138 字节**，SHA-256 **`00e093b43fecc115708568cc0e47c667863e6963ab1fa4d0147f8fbed5930aac`**。Test AUC 为 `null`，未访问隐藏标签。
- `cost.py --self-check-archive` 调用与 `predict.py` 相同的 `load_models()` 和 `predict_image()`，在预先按 ID 选定的 100 个单图 Test Study 上复推，最大概率差 **0**；全部 1,414 个 Study 的宽表与四列 Submission 差值 **0**。这不是主办方固定 100 张计时图，不能代替官方 T4 验收。
- 纯推理权重 SHA-256 **`35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69`**，81,651,999 字节，与朋友来源 manifest 匹配。权重加两份必要配置 **81,654,915 字节**；清单所列非结果文件 **81,876,207 字节**，实际当前目录所有非结果文件（含生成的 Python 缓存和清单）**82,261,823 字节**，均小于 500,000,000 字节。`package_manifest.json` 记录 63 个重要文件大小与 SHA-256；`SHA256SUMS.txt` 含清单本身。独立逐文件核验全部通过。
- 原冻结 E08 Submission SHA-256 仍为 `9a9846947a544e759cc19c79f6cc8c7fae39508058ee0e3d98d3deb24e8b2a07`；朋友原 H17 Submission SHA-256 为 `1bd7a6deae6d716b9ae4e08dd29d2cb1a83ae16b9d49c6fa016629810d88f30e`，均未覆盖。新候选单独保存。
- 本轮没有重复跑 4060 计时。Phase30 相同 batch=1 入口的本地参考均值为 **88.46ms**（另一次 93.77ms）；官方状态始终为 **T4_UNVERIFIED**，不能据此宣称满足 100ms。

## 待完成的官方合规项目

1. 真实 T4 16GB 上使用主办方固定 5 张预热、100 张正式图，batch=1 连续 3 轮；测从文件读取到结果写出的完整耗时，并和同入口新 Submission 核对。包内 README 给出命令；本机没有 T4，因此未执行。
2. 提供最终材料要求的训练代码、原始 ImageNet 预训练文件及其独立 SHA、权重再分发许可全文/上游 model card、核验 Excel、数据使用声明。现已附 torchvision 软件 BSD-3-Clause，但这不自动覆盖预训练权重的许可。朋友训练数据授权仍需人工确认。
3. 如正式部署需将图像加载方式换成评测平台提供的文件目录，应先通过同一 `predict_image(Path)` 路径重新验证新候选概率；不得沿用旧批量推理 CSV 冒充本包结果。

结论：**LOCAL_CONSISTENCY_PASS_T4_UNVERIFIED**。当前可用于本地候选审阅，尚不能标为官方部署 PASS；未训练、未上传排行榜、未 Push GitHub。
