# E08-EMA 部署审计

- 官方规则：https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf（第 1、7、8 条及附录 A）。主办方目标硬件为 T4 16GB；单图 batch=1、5 张预热、100 张正式、连续 3 轮，读图至写出最终结果。官方上限为 **≤100ms**。
- 冻结来源：`E:\计挑赛\cxr-multilabel-challenge\outputs\submissions\E08_EMA_20261020\manifest.json`；5 个推理 checkpoint 的文件长度和 SHA-256 与 manifest 逐一一致，5 份配置哈希也一致。完整权重 **278,436,014 字节**，连同现有配置 **278,442,617 字节**，低于 500,000,000 字节。仍需对最终材料目录全部必要文件重新计数。
- 公式：`E01=(B01+B02+B03)/3`; `E03=0.5*E01+0.5*M02`; `E08=0.5*E03+0.5*N01_EMA`。所有模型 384×384、`black_pad_v1`、ImageNet RGB 归一化、FP16 autocast、Study Mean。
- Phase20 的独立共享预处理入口在 `outputs/analysis/phase20_inference_optimization/optimized_inference/shared_preprocess.py`；其全量 1422 图/859 Study 检验在 `prediction_equivalence.json` 为 `accepted=true`、概率差 0。**该实现未并入正式 `scripts/predict.py`。** 本目录 `deployment/predict.py` 是独立副本，读图/预处理只做一次，五模型共同使用同一张量。
- 本次 4060 测速来自官方 **Val** 中 100 个不同单图 Study，原图文件读入到 CSV 写出；5 张另选的 Val 图预热、100 张图 × 3 轮。均值 **182.39ms**、P50 181.77ms、P95 200.07ms、最大 213.80ms。结果是本地参考，**T4_UNVERIFIED**；所用图片也不是主办方正式固定 Test 计时图。
- `cost.py` 直接调用 `predict.py` 的 `load_models()` 与 `predict_image()`。batch=1 图像输出对照冻结单图 Study 的 Val 概率最大差 **0.000793457**，高于事先采用的 1e-4 门槛；100/100 Study 完整参与比较。推测与原验证批量 8/16 和单张前向的数值计算路径有关，尚未通过同一预测一致性门禁。**不能宣称已达到最终提交文件一致性要求。**
- E08 冻结验证 Macro-AUC 0.811668646，官方排行榜 77.75（历史成绩，非本次测速得出）。正式部署仍需 T4 上以官方 5+100 图片执行 `cost.py`，并验证预测与最终 `submission.csv`；如采用不同精度/实现，还需全量 Val 概率与 AUC 复算。
