# E08 PyTorch FP16

独立5图 E08 固定融合，五个原始 checkpoint，384黑边填充、FP16 autocast、Study Mean。`predict.py` 包装 Phase30 PyTorch 入口，`cost.py` 调用同一入口。本次 batch=1 路径相对旧冻结 Val CSV 最大差0.000793，100个单图 Val Study 中有1个类别排序差异；旧 CSV 不在本包内，也不得把旧 CSV 的提交一致性标为通过。正式提交须由当前入口重新生成对应提交文件并在官方 Test 样本上验证。T4_UNVERIFIED。
