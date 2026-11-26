# H17-02 batch=1 T4 复测准备

部署包：`outputs/analysis/phase31_h17_deployment/H17_02_T4_candidate/`。本机复核其 `package_manifest.json` 列出的 63 个文件，缺失及哈希变更均为 0。复制到获授权的 T4 16GB 环境时保留原目录结构及文件名，使用兼容该环境驱动的 PyTorch/torchvision CUDA wheel；不得因安装需要改动模型、FP16、batch=1 或 384 输入。`requirements.txt` 记录本机版本，T4 兼容性需在目标机核实。

在目标机先核对 GPU、依赖版本、所有实际加载文件 SHA 和未压缩字节数。取得主办方固定的 `pictures_5`、`pictures_100` 后，从部署包目录执行（路径需填目标机实际位置）：

```bash
python write_test_image_manifest.py --config config/deployment_config.json --archive /authorized/official_test.zip --output /new_results/full_test_images.csv
python build_cost_study_index.py --config config/deployment_config.json /new_results/full_test_images.csv /authorized/pictures_100 /new_results/cost_study_index.csv
python cost.py --config config/deployment_config.json --submission /authorized/H17_02_BATCH1_20261024/submission.csv --pictures-100 /authorized/pictures_100 --pictures-5 /authorized/pictures_5 --study-index /new_results/cost_study_index.csv --official-sample --output-dir /new_results/cost_t4
```

`cost.py` 实际调用 `predict.py` 的 `load_models()` 和 `predict_image()`。它按每张 3 轮均值再对 100 张取均值，CUDA 同步并计入读图到写出结果的完整墙钟时间。仅在图片确由主办方固定提供时使用 `--official-sample`。核验输出 `cost_report.json` 的设备、精度、样本数、`mean_ms ≤ 100`、预测一致性和文件哈希；若计时样本只覆盖多图 Study 的一部分，报告应保留 `PARTIAL_SAMPLE` / `PREDICTION_UNVERIFIED`，需用完整官方 Test 图像进一步核对对应 Study Mean。不可用 4090/4060 结果替代 T4 测量。

当前状态：**SIZE_PASS / T4_UNVERIFIED**。用户确认的 77.7 是排行榜成绩，不是部署验收结果，也不是可计算的隐藏 Test AUC。
