"""Read-only device/runtime audit; target GPU checks are evaluated on server."""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch
    result = {"platform": platform.platform(), "python": sys.version.split()[0],
              "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name()
        result["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True, check=True)
        result["nvidia_smi"] = query.stdout.strip()
    except Exception as exc:
        result["nvidia_smi_error"] = str(exc)
    try:
        import onnxruntime as ort
        result["onnxruntime"] = ort.__version__
        result["onnx_providers"] = ort.get_available_providers()
        result["onnx_cuda_provider_available"] = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception as exc:
        result["onnx_error"] = str(exc)
    result["t4_device"] = "T4" in result.get("gpu_name", "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
