# Inference environment

## Organizer target

The official server has one NVIDIA T4 (16 GB), eight CPU cores, 32 GB RAM, at least 100 GB SSD, NVIDIA driver 570.117 or newer, CUDA 12.8 and cuDNN 9 in the container, Python 3.12, PyTorch 2.10.0, and torchvision 0.25.0. The organizer's recommended image is `crpi-4leqt7vljo7ryef.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:1.0`. Verify image access and its exact spelling with the organizer if the registry rejects the pull.

The preliminary round needs only `submission.csv`; this project does not build or upload an image. The host driver and GPU hardware cannot be configured by the project. Final-round model and code packaging awaits the organizer's notice.

## Local Windows development environment

The current workstation has an RTX 4060 Laptop GPU with 8 GB and driver 581.08; Docker is not installed. Its GPU and memory differ from the official T4 host. The project-local `.venv` has Python 3.12.4, torch 2.10.0+cu128, torchvision 0.25.0+cu128, CUDA runtime 12.8, and cuDNN 9. CUDA is available; a GPU DenseNet forward pass and torchvision CUDA NMS passed. This verifies local GPU functionality, not T4 latency or official host compliance.

The D: Anaconda Python bundles older Visual C++ runtime DLLs that caused `c10.dll` WinError 1114. The local `.venv/Lib/site-packages/sitecustomize.py` preloads newer Windows system DLLs before importing torch. Its committed template is `docs/windows_sitecustomize.py.txt`. Copy the template when recreating a local virtual environment from that Anaconda Python. The official Linux/T4 environment does not need this Windows workaround.

With user approval, the [official NVIDIA 581.08 notebook driver](https://www.nvidia.com/download/driverResults.aspx/252938/en-us/1000/) was installed on 2026-10-16. The downloaded package in the ignored `.wheels/` directory was verified as 895,258,064 bytes with a valid NVIDIA Corporation Authenticode signature and SHA-256 `99354507EA54E8EB032B08EC1774B609F0A29AF38D7A03B76E3E57FA94622B42`. The installer exited 0 with automatic reboot suppressed; `nvidia-smi` then reported 581.08 and PyTorch CUDA remained available. `check_environment.py --require-official-host` still fails locally because an RTX 4060 Laptop is not a T4.

The official CUDA 12.8 wheel channel and local setup commands are documented in README.md. On the T4 server, use the recommended preconfigured environment or the official CUDA 12.8 wheel channel. Install only `requirements/inference.txt` for deployment and `requirements/train.txt` for training. The checkpoint contains a `state_dict`, model name, class count, and format version; it contains no data or filesystem paths.

On the official server, use `--require-official-host` with `check_environment.py`. It requires PyTorch CUDA 12.8, cuDNN 9, a visible T4 with at least 15,000 MiB, and the declared driver minimum. Use `--require-cuda` for local software-stack checks. Run model latency tests on the actual T4 server for a meaningful comparison with the 100 ms limit.
