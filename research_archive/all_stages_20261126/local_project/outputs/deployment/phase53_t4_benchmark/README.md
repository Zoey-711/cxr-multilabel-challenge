# Phase53 T4 官方口径测速包

本包是四套相互隔离的**测速候选**。它不含比赛影像、隐藏 Test 标签或训练数据，也不代表已通过主办方复测。每套 `cost.py` 直接导入同目录 `predict.py`，并执行该候选实际模型加载与逐图前向。先运行 `verify_packages.py`，任何哈希不一致即停止。

## 规则依据及边界

项目未找到用户所列名称完全一致的《比赛参赛规则.pdf》《关于比赛参赛规则若干事项的补充说明.pdf》或《模型核验材料与数据申明清单模板(1).xlsx》。本地找到此前保存的 `official_rules_reference.pdf`，对应先前 Phase45 保存的官方规则 PDF；第7/8/10条的端到端计时、500MB、材料要求可在该文件中复核。补充说明原件缺失，特别是样本级判定/排序与 `1e-3` 的解释仍须用户对照原件确认。本脚本按用户给出的严格要求执行。

计时：独立5张预热；100张串行、batch=1、连续3轮；每张图的3轮均值再对100张取总体均值。计时从读取文件前开始，包含解码、预处理、前向、后处理、预测写入与 CUDA 同步；模型加载和预热不计。完整 300 行见 `timings_300.csv`，对应概率见 `predictions_300.csv`。`P50/P95` 为每张图三轮均值的分位数，仅作辅助。

Study Mean 的提交一致性需要**官方完整影像索引**。100张测速样本若只包含一个多图 Study 的部分图片，该 Study 不可拿单图概率与 Submission 比较；本脚本只比较索引中影像齐全的 Study。索引列严格为 `image,Subject_id,Study_id,study_image_count`，其中 `image` 是 100 张目录下的唯一文件名，`study_image_count` 是官方 Test 中该 Study 的全部影像数。提供当前推理版本对应的 `submission.csv` 与索引后，核验 0.5 判定、10类排序和绝对概率差 ≤1e-3。缺索引或缺对应提交文件时报告 `SUBMISSION_CONSISTENCY_UNVERIFIED`。

## Linux T4 环境

建议 Python 3.12、T4 16GB、系统内存至少16GB、可用磁盘至少12GB。这是为解压、四套模型、独立虚拟环境、临时输出与用户提供的105张图留的**建议余量**，不是赛事规定的硬件最小值；真实需求以云镜像和图像体积为准。先核验云镜像的驱动可运行与本地测试相同的 CUDA PyTorch 构建。不要自行降级驱动来适配软件。ONNX Runtime CUDA Provider 与 CUDA/cuDNN 是否匹配，由环境检查和候选模型加载共同验证。

```bash
tar -xzf phase53_t4_benchmark.tar.gz
cd phase53_t4_benchmark
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r candidates/H17_BATCH1_FP16/requirements.txt
python -m pip install -r candidates/H17_ONNX_FP32/requirements.txt
python verify_packages.py
python check_environment.py --output reports/t4_environment.json
```

上述 pin 为本机已验证版本（PyTorch 2.10.0、torchvision 0.25.0、ONNX Runtime GPU 1.23.2）。如果目标镜像的驱动、CUDA、cuDNN 不兼容，先选能在 T4 上实际运行的对应 CUDA wheel 环境，并重新运行全部一致性核验。`check_environment.py` 报告设备/驱动/运行时；各 ONNX 模型额外禁用 CPU 算子回退，CUDA Provider 不可用即失败。

## 指定图片与单候选测速

用户需把主办方提供的 `pictures_5/`、`pictures_100/` 及完整 Test 映射索引私下传到服务器。**不要**把这些图放进本包或公开地址。示例采用绝对路径，当前工作目录可任意选择：

```bash
PKG=/path/to/phase53_t4_benchmark
P5=/private/pictures_5
P100=/private/pictures_100
IDX=/private/timed_image_index.csv
python "$PKG/candidates/H17_BATCH1_FP16/cost.py" \
  --pictures-100 "$P100" --pictures-5 "$P5" \
  --submission "$PKG/candidates/H17_BATCH1_FP16/reference_submission.csv" \
  --image-index "$IDX" --official-sample --output "$PKG/reports/t4_H17_BATCH1_FP16"
```

其中 `--official-sample` 只在确认使用主办方固定样本时传入。要一键运行四套：

```bash
bash "$PKG/run_all.sh" --pictures-100 "$P100" --pictures-5 "$P5" \
  --image-index "$IDX" --official-sample --output "$PKG/reports/t4_all"
```

可用 `--candidate H17_ONNX_FP32` 只跑一套；可用 `--submission H17_ONNX_FP32=/private/new_submission.csv` 为无已提交参考的候选指定后来真实生成的对应 CSV。其他候选同样可用 `--submission CANDIDATE=/private/corresponding_submission.csv` 覆盖内置参考文件。不要用其他候选的 CSV。

测速结束后，下载 `reports/t4_all/` 内每套的 `report.json`、`timings_300.csv`、`predictions_300.csv` 与 `environment.json`，并保留本包 SHA-256 清单。`T4_MEASURED` 仅表示设备为 T4，最终仍需检查样本来源、提交一致性、均值≤100ms、许可证和主办方复测。

## 候选区别

| 候选 | 推理精度 | 对应官方成绩 | 本地 Val 样本核验 | 历史 Submission |
|---|---|---:|---|---|
| E08_ONNX_FP32 | ONNX CUDA FP32 | 77.74 | 100个单图 Study 与同版本 Val 完全一致 | 内置 |
| H17_BATCH1_FP16 | PyTorch FP16 | 77.70（用户确认） | 同版本 Val 完全一致 | 内置 |
| E08_PYTORCH_FP16 | PyTorch FP16 | 77.75（历史入口） | 相对旧冻结 Val 有数值/排序差异，待正式提交一致性核查 | 无当前入口匹配CSV |
| H17_ONNX_FP32 | ONNX CUDA FP32 | 尚无 | 同版本 Val 完全一致 | 无；提交一致性未验证 |

全部四套文件大小均低于500,000,000字节；详见各 `manifest_sha256.json`。四套同包的总大小不代表单一正式模型大小。所有本地4060结果仅作参考，T4 未实测。
