#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 2 || "$1" != "--config" ]]; then
  echo 'Usage: bash scripts/friend_remote_preflight.sh --config configs/friend_server.local.env' >&2; exit 2
fi
source "$2"
PLAN=configs/batch_plan.yaml
: "${CXR_DATA_ROOT:?Set authorized official ZIP path}"
: "${CXR_PROTOCOL_DIR:?Set authorized fixed protocol metadata directory}"
: "${CXR_OUTPUT_ROOT:?Set authorized result directory}"
: "${CXR_PRETRAINED_ROOT:?Set authorized ImageNet cache root}"
: "${CXR_BACKUP_ROOT:?Set authorized backup directory}"
: "${CXR_GPUS:?Set authorized GPU IDs}"
: "${CXR_MAX_CONCURRENT:?Set authorized concurrency}"
: "${CXR_PREFLIGHT_GPU_HOURS:?Set a separate friend-approved preflight GPU-hour cap}"
: "${CXR_PREFLIGHT_VERSION:?Set a unique preflight version, for example v1}"
[[ "$CXR_PREFLIGHT_VERSION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe preflight version' >&2; exit 2; }
if [[ ! -x .venv/bin/python ]]; then
  bash scripts/cloud_install.sh --config configs/batch17/d17_fold1.yaml
fi
source .venv/bin/activate
python scripts/protocol_assets.py --config "$PLAN" --action import --directory "$CXR_PROTOCOL_DIR"
python scripts/batch_run.py --config "$PLAN" --action validate
python scripts/remote_preflight.py --config "$PLAN" \
  --output "$CXR_OUTPUT_ROOT/phase17_batch_preflight_$CXR_PREFLIGHT_VERSION" --gpus "$CXR_GPUS" \
  --max-concurrent "$CXR_MAX_CONCURRENT" --budget-hours "$CXR_PREFLIGHT_GPU_HOURS"
echo "REMOTE_PREFLIGHT receipt: $CXR_OUTPUT_ROOT/phase17_batch_preflight_$CXR_PREFLIGHT_VERSION/receipt.json"
