# H17-02 合规与最终材料核对

依据[官方规则 PDF](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)及 Phase29 只读审计。本文件不把本机测试标记成官方通过。

| 项目 | 当前状态 |
|---|---|
| 推理权重 | `model/H17_02.pth`，纯 `state_dict`，81,651,999 字节，SHA-256 已与朋友 `H17_DISTILL_02` manifest 核对。 |
| 必要配置与代码 | `config/deployment_config.json`、`config/resolved_config.yaml`、`src/cxr/`、`predict.py`、`cost.py`、`requirements.txt` 已入独立包；逐文件哈希见 `package_manifest.json`。 |
| 500MB | 本包实际文件大小及全部加载模型的字节数将在最终清单中核算；须以未压缩大小判定。 |
| 100ms | **T4_UNVERIFIED**。Phase30 本机 RTX4060 batch=1 完整流程均值 88.46ms（另一次 93.77ms），只能作为参考。 |
| 预测一致性 | 新 batch=1 Submission 必须由本包产生；`cost.py` 自检及官方固定样本对照结果见 `results/`。冻结朋友 Submission 不能替代新文件。 |
| 训练数据 | 朋友 `run_manifest.json` 声明训练只用比赛官方 Train、官方 Val 选最佳，教师来自完整 OOF；朋友实际数据使用授权和最终数据声明仍需人工签署。没有访问隐藏 Test 标签。 |
| 初始预训练 | `torchvision EfficientNet_V2_S_Weights.IMAGENET1K_V1`，公开源 `https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth`，训练 manifest 记录 SHA-256 `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`。本机未取得原始预训练文件，尚不能独立复核该文件 SHA。 |
| 许可证 | 已附 torchvision 软件的 BSD-3-Clause 全文。软件许可不自动覆盖 ImageNet 预训练权重的再分发许可；权重上游 model card、训练数据说明及再分发授权仍需核验、附全文。 |
| 训练可复现材料 | 朋友训练源 commit `8b7944d744f91e753b912367adeb354e8b7e21d5`、完整训练脚本、原始预训练文件、核验 Excel 及数据声明仍需汇集；本包当前仅为推理候选。 |
| 历史最高分材料 | 现有官方最高分为 E08-EMA 77.75；正式最终材料若要求历史最高分 Submission，须另行按官方要求提供，不以 H17 候选冒充。 |

正式 T4 验收尚需：目标 T4 16GB、官方固定 5+100 张图片、batch=1 三轮读图至结果写出的计时、Study 映射和预测一致性检查，以及最终材料清单的人工审阅。任何一项缺失时不得标记为部署 PASS。
