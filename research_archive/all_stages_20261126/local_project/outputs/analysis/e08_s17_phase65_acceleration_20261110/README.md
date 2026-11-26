# 六模型固定融合：Phase65代码复用验证

固定公式：`0.375 E03 + 0.375 N01 EMA + 0.25 S17-F3`。384输入、六模型、FP16 autocast、Study Mean均保持不变。

实现位于项目 `src/cxr/e08_s17_acceleration.py`。本目录 `predict.py` 为独立本地入口，`cost.py` 实际调用它；没有替换任何历史推理文件。

从项目目录运行（输出目录必须不存在）：

```powershell
.\.venv\Scripts\python.exe outputs/analysis/e08_s17_phase65_acceleration_20261110/cost.py --config outputs/analysis/e08_s17_phase65_acceleration_20261110/config.json --output outputs/analysis/e08_s17_phase65_acceleration_20261110/new_cost_run
```

使用已冻结Val样本，5张独立预热、100张正式图、3轮，batch=1。单图计时包括读取、解码、预处理、GPU传输、全部前向、融合及图像预测CSV写入/flush与CUDA同步。模型加载、CUDA Graph捕获和预热不计时。Study Mean在完整Val核验中单独执行；上述耗时是单图端到端，不是多图Study总耗时。

优化为灰度图先缩放填充再转RGB、一次性回传六模型输出、固定输入缓冲区及CUDA Graph。非灰度图回退原预处理。每张图仍真实执行六模型前向，不缓存预测。

`run/full_val_consistency.json`记录1422图/859 Study完整验证。`run/`保存初次实验，`repeat_completed/`保存配对复测，`final_entry_cost/`保存最终入口测试。原进程在第一次重复测速中途结束；其`run/baseline_repeat/`不是完整可引用结果，以独立补跑为准。

此目录引用本机冻结模型和Phase53工具，是本地验证实现，并非自包含Linux部署包。T4迁移需打包依赖与模型、改为相对路径，在目标机器重新捕获CUDA Graph，并重新验证概率及100ms口径。没有使用F盘包里的T4 TensorRT引擎，没有进行本次T4测试。

优化与本次batch=1基准完全一致，不代表与78.81历史CSV逐值一致。历史不同batch产生的数值差异见报告；本次未生成Test预测或替换Submission。
