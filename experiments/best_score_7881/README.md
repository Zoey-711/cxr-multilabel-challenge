# 78.81最高分完整部署交付

官方成绩78.81由用户确认；Val Macro-AUC=0.8145689350883447。
固定公式：0.75 E08 + 0.25 本地S17-F3 = 0.375 E03 + 0.375 N01 EMA + 0.25 S17-F3。
六模型权重：B01/B02/B03各0.0625，M02为0.1875，N01 EMA为0.375，S17-F3为0.25。模型、预处理与融合参数不变。

## 获取完整工程

**当前发布状态：完整ZIP已在本地构建并核验，Release尚未发布。发现当前仓库公开，正在等待用户确认公开发布或指定私有仓库；下列Release下载命令在发布完成前不可用。本地已有ZIP可用--archive验证解压。**

本目录runtime_source提供完整部署源码供审阅；Release ZIP交付完整权重、T4引擎、配置、运行依赖模块、来源材料、历史submission.csv及Test图像映射。没有原始比赛图片或训练标签。Git源码目录不能代替Release全部文件。

需要仓库访问权限及GitHub CLI登录（不要在脚本中填写Token）。在仓库根目录执行：

```bash
python experiments/best_score_7881/fetch_deployment.py --destination /workspace/best7881
```

也可从Release手动下载best_score_7881_study_v5.zip，再执行：

```bash
python experiments/best_score_7881/fetch_deployment.py --archive /path/to/best_score_7881_study_v5.zip --destination /workspace/best7881
```

下载器锁定资产大小与SHA-256，并逐项验证351个文件，安全解压到新目录。解压后无需本地E盘或另一份私有源码。

## 运行环境及正式入口

使用官方镜像：crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0。
目标T4；Python3.12、PyTorch2.10.0 CUDA12.8/cuDNN9、TensorRT10.9.0.34、ORT GPU1.26.0。不要在官方镜像升级框架。requirements.txt注明沿用官方预装依赖；require_environment检查缺失依赖并报错。已提供引擎来自T4，不保证在不同GPU或TensorRT版本可复用；失败时不得CPU回退或伪造结果。

```bash
cd /workspace/best7881
python model/check_environment.py --config model/config.json --output /workspace/environment_check.json
python cost.py submission.csv /authorized/pictures_100 /authorized/pictures_5
```

最后一条严格三个位置参数。100个完整Study、另外5个Study预热、batch=1逐图串行、3轮。元数据包内加载；全部图片推理、Study Mean和最终终端写出计时；cost.py不产生结果文件。官方示例只有10个计时Study，不足正式100个，不能用它声称完成正式测速。图片需用户合法提供。

## 当前验证边界

完整工程文件/哈希验收通过，Study计时CPU模拟通过。没有因本次归档重新做T4实测。84.46ms来自历史100个单图Study的Val参考测试，不代表多图Study新版cost.py成绩。模型同源不等于数值完全相同：历史CSV包含旧批量FP16路径，部署为PyTorch FP16 CUDA Graph + B02/M02 TensorRT FP32；当前入口相对历史78.81 CSV的一致性及Study口径速度仍待目标环境验证。不声称官方合规已通过。

source保存历史CSV生成脚本，provenance保存来源及指标。历史N01 Test图像级概率未找到，因此不声称六模型图片均值全部独立复算。历史manifest生成状态保持原文，score.json单独记录用户确认成绩。

全部阶段路线与源码见../../research_archive/all_stages_20261126/README.md。
