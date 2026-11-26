# 六模型 batch=1 T4 候选交接

- 历史提交：`outputs/submissions/E08_075_S17_F3_025_20261110/submission.csv`，SHA-256 `8e2d50c806da5e033f2f649f1186e0c9856740cb2ba720cd42fb26c54e02772c`，官方 78.81（用户提供）。冻结文件未改动。
- 新候选包：`outputs/deployment/e08_s17_t4_batch1_candidate_20261110.tar.gz`，SHA-256 `501104c9ab5d504de20e4eb94dfacdda9960664f272d92260d4f1e1ece1f8b8f`，362,499,071 字节。
- 固定公式：`0.75 × E08 + 0.25 × S17-F3`。保持 E08 原五模型及权重，另加载 S17-F3；B01、B03、N01 EMA、S17-F3 为 PyTorch FP16 CUDA Graph，B02、M02 为原 Phase65 T4 FP32 TensorRT 引擎。所有模型每张图片均真实执行一次。
- 实际加载的模型、引擎及必要配置共 392,800,139 字节，低于 500,000,000 字节。压缩包体积不是部署大小判断依据。
- 本地检查：32 个文件 SHA-256 全部匹配；压缩包 33 个成员可读取且路径安全；S17-F3 在 CPU 上严格加载为 SwinTransformer；全部 Python 源码可解析；根目录 `predict.py` 与 `cost.py` 使用同一个 `load_models`、`predict_image` 函数；100 张图片索引完整性测试通过。
- 本机没有 TensorRT，因此未在 Windows 上执行混合引擎六模型推理，也没有六模型 T4 延迟。原 Phase65 五模型的 T4 测速及一致性结果不能替代六模型验收。历史 batch 推理精确复现 78.81 CSV，但该路径不符合 batch=1；本机全 PyTorch batch=1 Test 曾有 10 个 Study 类别排序变化，因此必须实际检查本包的混合路径。
- 状态：`T4_UNVERIFIED`、`HISTORICAL_SUBMISSION_RANKING_UNVERIFIED`。在目标 T4 上加载引擎，对官方规定样本执行 5 张独立预热、100 张计时、连续 3 轮，并核对与该历史提交对应的完整 Study Mean 预测。若图片只覆盖部分 Study，工具明确给出 `UNVERIFIED_PARTIAL_STUDY`，不能据此宣称一致。

运行命令与输入索引要求见部署包内 `README.md`。若六模型的概率、判定或排序不满足规则，保留历史提交不变，记录真实差异；不能通过改概率、融合比例或更换计时图片来制造通过结果。
