"""Check framework versions and optional CUDA availability without loading weights."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from cxr.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-official-host", action="store_true")
    args = parser.parse_args()
    load_config(args.config)
    try:
        import torch
        import torchvision
    except (ImportError, OSError) as exc:
        parser.exit(1, f"PyTorch import failed: {exc}\n")

    driver = None
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        driver = process.stdout.strip().splitlines()[0].strip()
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        pass
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_memory_mib = round(torch.cuda.get_device_properties(0).total_memory / 1048576) if torch.cuda.is_available() else None
    driver_ok = driver is not None and tuple(int(part) for part in driver.split(".")) >= (570, 117)
    official_host = bool(gpu_name and "T4" in gpu_name and gpu_memory_mib and gpu_memory_mib >= 15000 and driver_ok)
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": gpu_name,
        "gpu_memory_mib": gpu_memory_mib,
        "nvidia_driver": driver,
        "meets_declared_official_host": official_host,
    }
    print(json.dumps(info, indent=2))
    ok = sys.version_info[:2] == (3, 12)
    ok &= torch.__version__.split("+")[0] == "2.10.0"
    ok &= torchvision.__version__.split("+")[0] == "0.25.0"
    if args.require_cuda or args.require_official_host:
        cudnn = torch.backends.cudnn.version()
        ok &= torch.version.cuda == "12.8" and torch.cuda.is_available()
        ok &= cudnn is not None and cudnn // 10000 == 9
    if args.require_official_host:
        ok &= official_host
    if not ok:
        parser.exit(1, "Environment does not meet the requested software or host requirements.\n")


if __name__ == "__main__":
    main()
