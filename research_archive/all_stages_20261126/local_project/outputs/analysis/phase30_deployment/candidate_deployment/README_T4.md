# Phase30 候选部署入口（待 T4 验收）

本目录的 `predict.py` 与 `cost.py` 共享 `load_models()` 和 `predict_image()`；`cost.py` 实际导入并调用二者。这里保留原始 E08/H17 权重及固定融合，不改变 384×384、黑色保比例填充、ImageNet 归一化、FP16 autocast 或 Study Mean。当前本机 4060 测试不能证明 T4 合规。

## 准备

在含 `src/cxr/` 的项目根目录执行。先使用目标环境已有、互相兼容的 CUDA PyTorch 与 torchvision；以 `python -c 'import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name())'` 核验。需 Python、PyTorch、torchvision、NumPy、Pillow。环境中缺 NumPy/Pillow 时可运行 `python -m pip install -r outputs/analysis/phase30_deployment/candidate_deployment/requirements_t4.txt`。PyTorch/CUDA 应按目标 T4 驱动和现有镜像选择匹配版本，不为本说明升级驱动。当前本机版本见 `../environment.json`。

将已核对 SHA-256 的推理权重放在独立目录：

```text
model_files/B01.pt
model_files/B02.pt
model_files/B03.pt
model_files/M02.pt
model_files/N01_EMA.pt
model_files/H17_02.pth
```

E08 五个权重和 H17 权重均由 `predict.py` 在加载前核对 SHA-256；`frozen_e08_manifest.json` 是只读的原始来源清单。只测一个候选时只需放它所需的文件。Test 图像、完整无标签图像索引与提交文件由比赛方提供，均不属于本目录。

```bash
export CXR_PROJECT_ROOT="$PWD"
export CXR_DEPLOY_ASSET_ROOT="$PWD/model_files"
python -c 'import torch,torchvision; assert torch.cuda.is_available(); print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name())'
```

`full_test_images.csv` 必须每张图一行，字段为 `Subject_id,Study_id,image_path`，路径相对 `test_images/`；1414 个 Study，类别顺序始终为 1–10。使用同一推理入口生成**新的独立候选**，不可覆盖冻结 E08 Submission：

```bash
python outputs/analysis/phase30_deployment/candidate_deployment/predict.py --candidate E08_EMA --image-manifest full_test_images.csv --image-root test_images --output phase30_e08_submission.csv
```

官方固定的 5 张预热图与 100 张正式图放在 `pictures_5/`、`pictures_100/`。先以完整 Test 图像索引建立 Study 对照；若 100 张中某个多图 Study 只包含部分图片，脚本不会错误地拿单图概率与 Study Submission 比较：

```bash
python outputs/analysis/phase30_deployment/candidate_deployment/build_cost_study_index.py full_test_images.csv pictures_100 phase30_cost_study_index.csv
python outputs/analysis/phase30_deployment/candidate_deployment/cost.py phase30_e08_submission.csv pictures_100 pictures_5 --candidate E08_EMA --study-index phase30_cost_study_index.csv --official-sample
```

H17 使用相同两条推理/测速命令，但把 `--candidate` 改为 `H17_02`，提交文件名改为独立的 `phase30_h17_submission.csv`。只有图片确实是主办方固定样本时才能传 `--official-sample`。模型加载和 5 张预热不计入正式计时；正式 100 张按 batch=1 连续测 3 轮，每张从读文件到写出概率 CSV，CUDA 同步，用每张三轮均值再对 100 张取均值。`cost.py` 先校验提交 CSV 为 1414 Study × 10 类。只对 100 张样本中图像齐全的 Study 比较 Study Mean；还需按赛事最终验收方式确认全部样本的提交一致性。若 T4 设备、比赛指定的 100 张图片或提交对照未验证，报告保持 `LATENCY_UNVERIFIED`、`OFFICIAL_SAMPLE_UNVERIFIED` 或 `PREDICTION_UNVERIFIED`。

模型原始文件和必要配置的实际大小、SHA-256 见 `../model_size_and_hashes.json`。这份包只是候选：正式提交材料须包含全部实际加载的权重及必要配置，并遵守总大小 ≤500,000,000 字节与 T4 单张完整流程均值 ≤100ms。
