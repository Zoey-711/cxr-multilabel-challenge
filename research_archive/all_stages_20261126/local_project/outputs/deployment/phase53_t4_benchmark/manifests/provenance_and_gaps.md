# 模型来源与材料缺口

本包仅用于 T4 推理测速，不把训练教师、原始 ImageNet 初始化权重或比赛影像作为运行时依赖。四套候选的实际模型文件 SHA-256 与字节数见各候选 `manifest_sha256.json`。权重/ONNX 模型均从本机冻结产物逐项核对后复制，不做重新训练、转换或参数修改。

| 架构 | 公开 ImageNet 初始化 | 本机原始文件 SHA-256 | 推理用途 |
|---|---|---|---|
| DenseNet-121 | torchvision IMAGENET1K_V1；https://download.pytorch.org/models/densenet121-a639ec97.pth | a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395 | E08 B01/B02/B03 的训练初始化 |
| ConvNeXt-Tiny | torchvision IMAGENET1K_V1；https://download.pytorch.org/models/convnext_tiny-983f1562.pth | 983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d | E08 M02 的训练初始化 |
| EfficientNetV2-S | torchvision IMAGENET1K_V1；https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth | dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74 | E08 N01 EMA 与 H17 学生的训练初始化 |

上述 SHA 来自项目训练来源记录与本机缓存；打包时没有把这些原始预训练文件再装进测速包。`licenses/torchvision_LICENSE_BSD3.txt` 是 torchvision **软件**许可证全文，SHA-256 为 `c06363f9d33627dc5173b13522444cc85fbb4739f63e16d4587d5c0a165b5b1e`。该软件许可不能自动代替每个模型权重及上游 ImageNet 数据的赛事使用、再分发许可证明。

H17_DISTILL_02 的推理仅加载一个学生模型。其训练来源：朋友结果 commit `cbfc5997438f371086cd1fbe232890590cd67ffb`，训练代码 commit `8b7944d744f91e753b912367adeb354e8b7e21d5`；H17 使用 Train 患者上的真实标签及多骨干折外教师概率蒸馏，教师为 DenseNet、ConvNeXt、Swin 五折模型，教师来源与患者隔离审计应随正式材料提交。教师模型本身不在 H17 推理图中，故不计入 H17 运行时500MB，但不能在材料申报中遗漏。朋友实际训练数据使用/共享授权仍需人工核实。

**正式材料待补：**用户指定名称的两份规则 PDF 与 Excel 模板原件未在项目中找到；`official_rules_reference.pdf` 为此前 Phase45 保存的另一命名的官方规则副本。补充说明原件第九部分仍需人工核对。还需要上游预训练权重适用的许可全文与训练数据说明、H17 教师生成脚本/来源、朋友数据使用许可声明、填写并签署的模型核验 Excel，以及 T4 上的官方样本耗时与对应 Submission 一致性结果。没有这些材料，测速包不能被称为最终正式合规材料包。
