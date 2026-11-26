# Phase49 environment and verification

- Local GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 581.08.
- Python 3.12.4; PyTorch 2.10.0+cu128; torchvision 0.25.0+cu128;
  Pillow 12.3.0; NumPy 2.5.2. The existing isolated ONNX Runtime GPU is 1.23.2.
- CUDA Execution Provider was actually loaded. In a single profiled FP32
  graph pass, 400 recorded nodes were assigned to CUDA and none to CPU.
  The profile includes initialization and its operator durations are not
  representative steady-state latency; see `onnx_operator_profile.json`.
- TensorRT Python module, `trtexec`, and loadable `nvinfer` DLLs were absent.
  An ORT provider appearing in the available-provider list does not prove a
  working native TensorRT installation. TensorRT FP16 and FP32 are BLOCKED.
- No official T4 environment was available. The target's driver, CUDA 12.8,
  cuDNN 9 and ONNX Runtime GPU 1.23.2 combination must be checked on that host.
  See the [official ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
- Independent deployment FP32 entry reproduced the evaluation entry exactly
  for 1,422 Val images and 859 Studies; see `deployment_entry_consistency.json`.
  The 300 predictions from its separate `cost.py` exactly matched the original
  Phase49 FP32 benchmark CSV by SHA-256.
- Fixed-sample timing was repeated sequentially, not interleaved. Baseline
  means 89.62 and 90.61 ms give a limited check on run-to-run variation;
  thermal state and filesystem cache remain potential confounders.
