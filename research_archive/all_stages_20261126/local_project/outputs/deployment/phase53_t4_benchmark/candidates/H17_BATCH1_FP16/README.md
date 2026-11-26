# H17 BATCH1 FP16

独立 H17_DISTILL_02 学生，384黑边填充，ImageNet归一化，batch=1 FP16 autocast，Study Mean。`model/H17_02.pth` 哈希与冻结版本一致，`predict.py` 包装 Phase31 原始推理实现，`cost.py` 调用同一入口。`reference_submission.csv` 为 H17_02_BATCH1_20261024 版本。教师不参与推理，但最终材料需要单独说明教师OOF来源、训练代码、公开预训练来源与数据许可。T4_UNVERIFIED。
