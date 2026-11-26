#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp
export TMPDIR="$PWD/.tmp"
export PIP_NO_CACHE_DIR=1
if [[ "$#" -ne 2 || "$1" != "--config" || ! -f "$2" ]]; then
  echo 'Usage: bash scripts/cloud_install.sh --config configs/f17_densenet121_fold1_384.yaml' >&2
  exit 2
fi
PYTHON_BIN="${CXR_PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3.12 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo 'Python 3.12 not found; inspect the base image before installing' >&2
  exit 1
fi
"$PYTHON_BIN" -m venv --system-site-packages .venv
source .venv/bin/activate
if [[ "${CXR_INSTALL_FRAMEWORK:-0}" == "1" ]]; then
  python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
fi
python - <<'PY'
import sys, torch, torchvision
if sys.version_info[:2] != (3, 12): raise SystemExit('Python 3.12 required')
if torch.__version__.split('+')[0] != '2.10.0' or torchvision.__version__.split('+')[0] != '0.25.0':
    raise SystemExit('Base image must contain torch 2.10.0 and torchvision 0.25.0; use the approved CUDA 12.8 image or install the official cu128 wheels explicitly')
if torch.version.cuda != '12.8' or not torch.cuda.is_available():
    raise SystemExit(f'CUDA 12.8 GPU required; found torch CUDA={torch.version.cuda}, available={torch.cuda.is_available()}')
print('Base framework:', torch.__version__, torchvision.__version__, 'CUDA', torch.version.cuda)
PY
python -m pip install --no-cache-dir -r requirements/cloud.txt
python -m pip install --no-cache-dir --no-deps -e .
python -c 'import cxr; print("cxr package installed")'
