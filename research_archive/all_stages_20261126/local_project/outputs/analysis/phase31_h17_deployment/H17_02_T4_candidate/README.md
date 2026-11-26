# H17-02 独立 T4 候选部署包

本包只包含 H17-02（EfficientNetV2-S）的一份最佳推理权重、冻结配置、必要源码与核验脚本。模型输入 384×384、`black_pad_v1`、ImageNet 归一化、FP16 autocast、batch=1；多图 Study 对逐图 sigmoid 概率取均值。`predict.py` 和 `cost.py` **直接调用同一个** `predict_image()` 与 `load_models()`。没有任何 Test 标签，也不包含原始胸片。

## 本地已执行

从本包目录运行：

```powershell
python predict.py --config config/deployment_config.json --archive "F:\数据集\cxr_first.zip" --output-dir results
python cost.py --config config/deployment_config.json --self-check-archive "F:\数据集\cxr_first.zip" --submission results/submission.csv
```

第一条只读取官方无标签 Test 图像与元数据，并生成 `results/test_predictions.csv`（1414 Study × 10 概率）、`results/submission.csv`（14140 行）、`results/prediction_manifest.json`。第二条在按 ID 事先确定的 100 个单图 Test Study 上重复调用**相同推理函数**，核对 Submission；同时逐行核对全部 1414 Study 的宽表与四列表。报告为 `results/internal_consistency.json`。两个入口均拒绝覆盖已有正式结果。

## 在 T4 上复现

把本目录完整复制到有权使用比赛数据的目标机器；模型、配置、源码和许可证须保留原样。使用 Python 3.12、可在 T4 驱动上运行的相互兼容 PyTorch/torchvision CUDA wheel；`requirements.txt` 记录本机版本。先检查 `python -c 'import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name())'`。不得把本机 4060 的结果视为 T4 验收。

```bash
python -m pip install -r requirements.txt
python predict.py --config config/deployment_config.json --archive /absolute/path/to/official_test.zip --output-dir results_t4
python cost.py --config config/deployment_config.json --self-check-archive /absolute/path/to/official_test.zip --submission results_t4/submission.csv
python write_test_image_manifest.py --config config/deployment_config.json --archive /absolute/path/to/official_test.zip --output results_t4/full_test_images.csv
python build_cost_study_index.py --config config/deployment_config.json results_t4/full_test_images.csv /absolute/path/to/organizer_fixed_100 results_t4/cost_study_index.csv
python cost.py --config config/deployment_config.json --submission results_t4/submission.csv --pictures-100 /absolute/path/to/organizer_fixed_100 --pictures-5 /absolute/path/to/organizer_fixed_5 --study-index results_t4/cost_study_index.csv --official-sample --output-dir results_t4/cost
```

`--official-sample` 只可用于主办方固定图片。正式计时：模型加载和 5 张预热不计入；100 张图片 batch=1 连续 3 轮，每张从文件读取到写出最终概率，CUDA 同步；每张先取三轮均值，再对 100 张取均值。`cost_study_index.csv` 的字段为 `image,Subject_id,Study_id,study_image_count`，`image` 是相对计时图片目录的路径。只有 100 张计时图完整覆盖某个 Study 的全部图片时，才可与 Study Submission 直接比较。若缺少 T4、官方固定图片或完整 Study 对照，状态保持 `T4_UNVERIFIED` / `PREDICTION_UNVERIFIED`。

实际加载权重大小和逐文件 SHA-256 见 `package_manifest.json`；原权重 SHA-256 为 `35fb17621d21b1f0825bd3dd8b2090c2e0115593d3ef8b7b0cedc2a380022f69`。新候选 Submission 与冻结 H17/E08 Submission 是不同文件；不可用旧文件替代本包 batch=1 输出，也不可自动上传比赛平台。

官方材料核对及仍待人工确认的许可证/数据来源事项见 `COMPLIANCE.md`。本包是**待 T4 验收候选**，不是已通过 100ms 硬约束的正式材料包。
