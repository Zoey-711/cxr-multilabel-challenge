# 六模型T4 batch=1候选：待目标机验收

本包基于已核验的Phase65五模型T4源码，新增冻结S17-F3作为第六模型。公式固定为`0.375 E03 + 0.375 N01_EMA + 0.25 S17_F3`。Phase65原五模型的独立T4参考测速在其随机100张Test样本上为76.10ms且一致性PASS，**这些数字不能用于本六模型**。

六模型正式推理每次处理一张384×384图片，B01/B03/N01 EMA/S17-F3用PyTorch FP16 CUDA Graph，B02/M02调用原Phase65在T4构建的FP32 TensorRT引擎。六模型权重、引擎与必要配置合计392800139字节，小于500000000字节。只保存实际加载的三份E08 PyTorch权重、两份TRT引擎及S17权重；没有把未使用的B02/M02 PyTorch权重误算进部署包。TensorRT引擎可能与新的T4驱动或运行时不兼容，需要在目标T4实际加载核验。

`phase65_source/predict.py`是最终单图模型加载和推理实现；根目录`predict.py`导入同一个实现。根目录`cost.py`通过`cxr.e08_s17_t4_cost`导入并调用同一`phase65_source.predict`的`load_models`与`predict_image`。每轮100张逐张运行，共3轮；计时包括读图、解码、预处理、全部六模型前向、融合、图像概率写出/flush、完整Study的均值与Study概率写出、CUDA同步。模型加载、图捕获及5张独立预热图片不计时。

## 在T4上运行

将压缩包上传到私有T4服务器，解压，进入解压后的目录。**不要把比赛图像上传到公开仓库**。

```bash
tar -xzf e08_s17_t4_batch1_candidate_20261110.tar.gz
cd e08_s17_t4_batch1_candidate_20261110
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python check_environment.py
python cost.py --config config.json \
  --pictures-100 /private/path/pictures_100 \
  --pictures-5 /private/path/pictures_5 \
  --image-index /private/path/timed_image_index.csv \
  --submission submission_original_78_81.csv \
  --output reports/t4_run_01 --official-sample
```

`--official-sample`仅在这100张图片确由主办方提供时使用；本地抽样不要加。Python/PyTorch/TensorRT版本参照`requirements-lock.txt`及原Phase65环境（Python3.13.5、PyTorch2.8.0+cu128、torchvision0.23.0+cu128、TensorRT11.4.0.106）。若安装或引擎加载失败，不要悄然切到CPU或换精度；记录报错后在目标T4环境重新构建并单独核验引擎。

`timed_image_index.csv`严格四列：`image,Subject_id,Study_id,study_image_count`。若某Study只抽中部分图像，本工具仍保存300条真实计时，但提交一致性状态为`UNVERIFIED_PARTIAL_STUDY`，不会把单图概率与Study Mean CSV硬比，也不会宣称PASS。须依据主办方指定的图像到Study对应方法解决；不得为了通过校验替换官方100张样本。完整Study时，每轮核对全部类别的概率差、0.5判定和排序；任一失败即退出码2。`report.json`保留实际失败证据。

这是**预备验收包**：本地Windows无TensorRT，完成了语法、模型哈希、公式、batch=1循环和体积检查，未运行六模型T4推理，未验证其与78.81历史CSV在T4上对应。历史批量推理可精确复现CSV，但它不是官方batch=1入口；本机纯PyTorch batch=1全Test有10个Study排序变化。因此最终合规仍以目标T4、官方固定100张样本、官方接口和主办方复测为准。不能把此包状态称为已满足100ms或预测一致性通过。
