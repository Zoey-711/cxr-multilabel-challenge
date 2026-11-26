# Phase34：DenseNet-121 训练期辅助深层监督

唯一成对实验：`AUX-BASE` 与 `AUX-01`，均使用同一份合法 ImageNet 初始化和 seed42。训练仅使用冻结患者 Fold2–5（8844患者、14826 Study、24332图），Fold1（2211患者、3687 Study、6012图）仅作固定最后一轮的最终主输出评估。H17-02 是完整 Train 的 EfficientNetV2-S 蒸馏学生，不能用于此处的 Fold1 初始化；本试验不加载它，也不加载既有Phase17已训练权重。

两组固定5轮，不通过官方Val或Fold1选择Epoch。训练使用原B01的384黑边填充、ImageNet归一化、BCE、无增强、AdamW backbone LR 1e-4/head LR 3e-4、WD1e-4、2轮warmup+cosine。微批4×累积4，有效batch16，AMP；两组主模型初值和每轮训练图像顺序必须相同。AUX在denseblock3（倒数第二个DenseBlock，1024通道）接Global Average Pool+Linear(1024,10)，损失为主BCE+0.2×辅助BCE。辅助梯度可回传此前骨干，推理时删除辅助头。

本批是全新固定预算对照，不是复现Phase17按官方Val早停选模型的成绩。5轮是预先固定的本地Pilot计算预算，不保证网络已充分收敛。Fold1此前已有多轮研究使用，结果仍为EXPLORATORY/PARTIAL_OOF，不等于独立测试或完整五折。

朋友Phase32的8项是ConvNeXt-Small/EfficientNet系列学习率、分辨率、损失和增强；E08-Tune是EfficientNetV2-S EMA参数实验。此次DenseNet辅助监督不在这两队列中。本地已有H17分类头微调和ConvNeXt多视角头实验不等同于本试验。

从项目根目录执行（不自动推送或提交比赛）：

```powershell
.\.venv\Scripts\python.exe scripts/run_phase34_aux.py --config configs/phase34_aux_pair.yaml --action preflight
.\.venv\Scripts\python.exe scripts/run_phase34_aux.py --config configs/phase34_aux_pair.yaml
.\.venv\Scripts\python.exe scripts/run_phase34_aux.py --config configs/phase34_aux_pair.yaml --action status
```

正式入口先核验配置、源代码和划分哈希及CPU/CUDA预检，顺序运行BASE再AUX。每轮完整保存模型、optimizer、scheduler、GradScaler和所有RNG状态。进程异常后重复同一命令，只从最近完整Epoch恢复；已有DONE任务先验哈希后跳过。失败不自动更改batch/超参数，不从头偷偷重训。

每组输出：实时status.json、train.log、train_history.csv、每轮epoch_*.pt、last_training.pt、最终纯主网络inference_model.pth、Fold1 oof_predictions.csv、导出后独立复算exported_oof_predictions.csv、完整OOF输出一致性、逐类AUC、模型字节数、4060参考读图至CSV写出延迟和SHA-256/DONE。延迟5图预热+20图×3轮，仅为本地参考，状态T4_UNVERIFIED；不据此声称官方100ms合规。

两组真正完成后自动计算2211患者配对Bootstrap（2000次有效抽样，seed42），绘制主BCE训练曲线并生成final_decision.md。预先定义后续研究门槛：OOF Macro-AUC增加至少0.002、差值95%CI下界>0、单类下降不超过0.02。即使满足也不自动运行后四折。没有完成前，AUC差值、逐类变化及扩展建议保持PENDING。
