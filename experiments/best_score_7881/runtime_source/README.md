# 王卓颖最终提交材料：Study口径V5

历史最高分78.81对应submission.csv、全部模型、融合权重和冻结predict入口保持不变。

## 官方执行
在公告指定环境中使用已安装框架；附加依赖见requirements.txt，不升级CUDA/PyTorch/TensorRT。

```bash
python cost.py submission.csv pictures_100 pictures_5
```

参数严格按上述顺序。pictures_100包含100个完整Study，pictures_5包含另外5个完整Study；图片总数可超过100和5。递归读取jpg/jpeg/png，保留原始Test文件名。Study与图像的关系由model/test_image_index.csv加载，不增加参数。正式数据示例尚未提供，当前支持任意目录嵌套，依据真实唯一文件名匹配；不猜测未知文件名，不静默忽略缺图。

每个Study的全部图像逐张调用与predict.py相同的固定六模型推理入口，再取Study Mean。固定线性融合与Mean的数学顺序可交换，实际数值仍以CSV一致性检查为准，不声称浮点逐位等价。单Study计时包括图片读取、解码、预处理、传输、推理、聚合、最终概率终端写出和CUDA同步；模型加载、首次初始化、5个Study预热不计时。100个Study连续3轮，先逐Study取3轮均值，再取100个均值。

cost.py仅输出日志，不创建结果文件。最终打印官方[CONCLUSION]格式，真实报告耗时、不一致Study数量和混合精度。三轮均比较历史CSV，任一轮失败的Study计一次。保留原核验的0.5判定阈值、10类降序排序和1e-3概率误差门槛，不使用CSV概率补全预测。

84.46ms仅为旧T4上100个单图Study的Val参考测试，不是V5 cost.py含多图Study的复测成绩。官方最终结果以其指定样本复测为准。V5仅做CPU模拟测试，未重新运行GPU测速；不承诺概率一致性或100ms已通过。

materials及model内旧报告/旧代码保留用于来源追溯，不作为当前执行说明。官方入口仅为包根cost.py；不要执行model内旧cost.py。
