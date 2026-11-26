# 预训练权重来源与许可核验

官方 PDF 第 1、2 条要求非医学开源权重、可用于比赛并允许再分发，最终提供公开链接、许可证全文、上游数据说明及 SHA-256：https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf。

| 架构 | torchvision 权重 | 原始公开 URL | 训练 manifest 记录 SHA-256 | 本地原始权重文件复核 |
|---|---|---|---|---|
| DenseNet-121 | IMAGENET1K_V1 | https://download.pytorch.org/models/densenet121-a639ec97.pth | `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` | PENDING：原始权重未定位 |
| ConvNeXt-Tiny | IMAGENET1K_V1 | https://download.pytorch.org/models/convnext_tiny-983f1562.pth | `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` | PENDING：原始权重未定位 |
| EfficientNetV2-S | IMAGENET1K_V1 | https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth | `dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74` | PENDING：原始权重未定位 |

来源：本地 B01/M02/N01 的 `pretrained_source.json`，朋友 H17-02 的 `run_manifest.json`。torchvision 官方分类权重表注明这三种 ImageNet-1K 权重：https://docs.pytorch.org/vision/main/models 。torchvision 0.25.0 包内 BSD-3-Clause 全文已复制为 `torchvision_LICENSE_BSD3.txt`（SHA-256 `c06363f9d33627dc5173b13522444cc85fbb4739f63e16d4587d5c0a165b5b1e`），官方仓库许可：https://github.com/pytorch/vision/blob/main/LICENSE 。**软件仓库许可证不自动证明每份预训练权重的再分发权利；最终材料需核对权重适用许可及上游 model card/训练 recipe，并附全文。**

推理 checkpoint 的冻结 SHA 已核实；原始 ImageNet checkpoint 在本机未定位，因此这里只能确认训练记录所填哈希，不能声称完成原始预训练文件复核。朋友的额外医学数据/授权使用情况标记 MANUAL_CONFIRMATION_REQUIRED。
