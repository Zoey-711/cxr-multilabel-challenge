# H17-02 ONNX CUDA FP32 T4 validation candidate

This directory is an independent inference candidate. It does not replace the
frozen H17-02 FP16 submission or its original checkpoint. The ONNX model comes
from that checkpoint (source SHA-256 in the Phase49 export manifest).

Install dependencies in a fresh Linux environment with a CUDA/PyTorch wheel
compatible with the server driver. The listed versions reproduce the local
software APIs; compatibility with the actual T4 driver must be checked there.
ONNX Runtime's CUDA provider documentation lists CUDA 12.8 and cuDNN 9 for
the 1.21–1.26 GPU package series. Verify the target driver and its libraries
before installation: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
Do not fall back to CPU if `CUDAExecutionProvider` is absent.

Use the organizer's fixed five warmup and 100 measured single images. From
this directory:

```bash
python cost.py --warmup-dir /authorized/pictures_5 --measure-dir /authorized/pictures_100 --official-sample --output-dir /new_results/h17_onnx_fp32_cost_t4
```

The model is loaded before timing. Each image is processed individually,
three rounds, from file read through CSV result write, with CUDA synchronization.
`cost.py` imports `load_model` and `predict_image` from this directory's
`predict.py`. The report intentionally remains
`T4_MEASURED_PREDICTION_CHECK_PENDING` until full Test predictions generated
with this same entry are matched to a newly generated submission. Never use
the older frozen FP16 CSV as the ONNX candidate's output.
The helper functions provide Study Mean and CSV writing, but the official
Test image mapping and template validation have not been exercised for this
new candidate; its submission workflow remains PENDING.

Check the uncompressed model plus config/code bytes, ONNX Runtime CUDA node
placement, driver compatibility, and the organizer's complete deployment
requirements before asserting compliance. The 4060 timing is reference only.
