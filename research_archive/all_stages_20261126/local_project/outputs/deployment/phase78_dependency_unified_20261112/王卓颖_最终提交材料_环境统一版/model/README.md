# 王卓颖：官方 compete:3.0 六模型构建与验收包

当前状态：PREPARED_PENDING_T4_BUILD。本包不是已经通过官方验收的成品。
历史78.81的 submission.csv 字节不变。固定融合仍为 0.75×E08+0.25×S17-F3；
E08=0.5×E03+0.5×N01_EMA，E03=0.5×E01+0.5×M02，E01=(B01+B02+B03)/3。
输入384、batch=1、原预处理及Study Mean不变。未训练、未改融合权重、未修改概率。

## 本次修订

- 不再安装旧 requirements-lock.txt。官方镜像预装框架是唯一目标；代码记录并检查实际版本。
- 排除旧 TensorRT 11.4 引擎，提供原始FP32 ONNX、checkpoint与SHA来源，
  在官方TensorRT 10.9/T4中构建两个FP32引擎。重建不改变训练权重，但可能改变数值，必须复验。
- 四个PyTorch模型使用原FP16 CUDA Graph；B02/M02使用FP32引擎；没有CPU回退或备用精度。
- cost.py、完整Test审计直接导入根目录predict.py，使用完全相同load_models/predict_image。
- 构建目录与推理目录分离。正式推理目录没有构建用ONNX及多余checkpoint。
- 所有命令拒绝覆盖已有输出。依赖检查不以“发现T4”代替版本/驱动检查。

## 环境

主办方镜像（用户提供）：
`crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0`

Linux、Python3.12、PyTorch2.10.0 CUDA12.8/cuDNN9、TensorRT10.9.0.34、
ONNX Runtime GPU1.26.0、单张T4 16GB、驱动>=570.117、8CPU、32GB内存、>=100GB SSD。
TensorRT模块若只显示10.9.0，报告保留该值；镜像摘要和pip包记录用于核实构建版本。
没有在本机运行Docker/T4；Windows CPU测试不等于Linux或GPU验收。
requirements.txt为无安装操作的说明文件。禁止pip升级/降级镜像内框架。

## 在Linux T4服务器执行

假设已将本ZIP解压，目录命名为 `/work/official_kit`。另建输出目录：

```bash
mkdir -p /work/official_runs
docker pull crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0
docker image inspect crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0 > /work/official_runs/docker_image_inspect.json
docker run --rm --gpus '"device=0"' --cpus=8 --memory=32g --shm-size=2g \
  --entrypoint /bin/bash \
  -v /work/official_kit:/kit:ro -v /work/official_runs:/runs \
  -v /private/test_images:/images:ro \
  -v /private/official_100:/pictures_100:ro -v /private/official_5:/pictures_5:ro \
  crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0
```

`/private/...`必须替换为服务器实际目录，图片不在包内。官方100图和5张预热图不可自行替换。
容器内按顺序执行（路径均为绝对路径，可以从任意工作目录执行）：

```bash
python /kit/check_environment.py --config /kit/config.json --output /runs/environment.json
python -m pip freeze > /runs/pip_freeze.txt
python /kit/build_engines.py --config /kit/config.json --output /runs/candidate_trt109
python /runs/candidate_trt109/check_environment.py --config /runs/candidate_trt109/config.json --load-models --output /runs/model_load.json
python /runs/candidate_trt109/full_test_audit.py --images-root /images --output /runs/full_test
python /runs/candidate_trt109/cost.py /runs/candidate_trt109/submission.csv /pictures_100 /pictures_5 --output /runs/cost_01 --official-sample
```

构建只在第一次运行，耗时待T4实测，不计入推理计时。两项顺序构建，workspace每项2GiB；
关闭FP16/INT8/TF32构建标志。原有四模型FP16设置保持不变。
构建失败保留BUILD_FAILED.json，必须另选新输出目录重试，不能把半成品当成候选。
首次构建的候选目录会生成新的manifest及真实文件大小报告。

## 一致性与计时

完整Test审计针对冻结的2218图、1414 Study、14140概率，校验原图哈希。
它不修改submission.csv。cost每图重新执行六模型，5独立预热、100图连续3轮，
CUDA同步并计入读取、预处理、推理、写出/flush，以及可完成Study的Mean和写出。
先计算每图3轮均值，再计算100图均值；模型加载和预热不计时。

cost支持官方示例三个位置参数，也支持--submission/--pictures-100/--pictures-5。
默认依据包内真实完整Test映射自动生成100图索引；Val模拟需另给--image-index，
但Val模拟不能与Test submission核验，不能冒充正式验收。
如果官方100张未包含某Study全部影像，保持UNVERIFIED_PARTIAL_STUDY，
不缓存或提前推理缺失影像来加速计时，不擅自改变主办方样本。
须向主办方确认逐图片核验与Study Mean的接口关系。
0.5是沿用的内部二值检查约定，须确认官方Pred_Label/Gate定义；未确认不得宣布官方合规。
主办方样本级判定/排序不一致数必须为0，且概率误差<=1e-3。

## 验收顺序与禁止事项

环境检查通过→新引擎构建和实际加载通过→全Test一致性通过→官方100图测速与逐样本一致性。
如果全Test失败，停止将此版本作为78.81提交的部署对应物，不修改概率、阈值或融合权重补救。
新引擎未生成前不报告最终模型大小；生成后实际加载资产及必要代码/配置合计<=500000000字节。
本包包含构建源，因此体积不代表正式推理目录大小。不能把本ZIP直接当最终验收成品。
没有T4结果前：T4_UNVERIFIED；没有一致性结果前：HISTORICAL_SUBMISSION_UNVERIFIED。

## 仍需补齐

原始ImageNet预训练文件、再分发许可证明和上游数据说明；队伍签署声明；
主办方更新版Excel模板与当前模板差异核对（本地目前只有旧模板）；
官方Gate及部分Study核验口径；T4最终报告。历史证据在materials/history，只作历史记录。
训练代码、模型配置和历史来源仍随包提供。包内无原始医学图像、无Test标签、无凭据。
