#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 2 || "$1" != "--config" || ! -f "$2" ]]; then
  echo 'Usage: bash scripts/friend_run_all.sh --config configs/friend_server.local.env' >&2
  exit 2
fi
source "$2"
: "${CXR_DATA_ROOT:?Set official ZIP path}"
: "${CXR_PROTOCOL_DIR:?Set frozen protocol directory}"
: "${CXR_OUTPUT_ROOT:?Set authorized output directory}"
: "${CXR_PRETRAINED_ROOT:?Set authorized ImageNet weight directory}"
: "${CXR_BACKUP_ROOT:?Set authorized backup directory}"
: "${CXR_GPUS:?Set explicitly authorized GPU IDs}"
: "${CXR_MAX_CONCURRENT:?Set authorized concurrency}"
: "${CXR_PREFLIGHT_GPU_HOURS:?Set approved preflight GPU-hour cap}"
: "${CXR_GPU_HOURS:?Set approved total 26-task GPU-hour cap}"
: "${CXR_GH_GPU_HOURS:?Set approved G/H GPU-hour subbudget}"
: "${CXR_PREFLIGHT_VERSION:?Set preflight version}"
if [[ ! -x .venv/bin/python ]]; then
  bash scripts/cloud_install.sh --config configs/batch17/d17_fold1.yaml
fi
source .venv/bin/activate
python scripts/friend_one_launch.py --config configs/batch_plan.yaml
