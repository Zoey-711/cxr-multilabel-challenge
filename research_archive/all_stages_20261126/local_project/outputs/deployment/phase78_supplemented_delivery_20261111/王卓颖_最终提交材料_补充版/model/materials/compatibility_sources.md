# 环境来源与兼容性

官方镜像及版本来自用户2026-11-10提供的服务器参数，见official_environment.json。
TensorRT引擎不能假定可由较旧运行库加载，因此旧11.4引擎不进入修订包。
NVIDIA官方说明：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
本包没有把最新版文档当作TensorRT10.9实测证明；10.9构建尚待官方镜像执行。
