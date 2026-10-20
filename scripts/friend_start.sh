#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 2 || "$1" != "--config" ]]; then
  echo 'Usage: bash scripts/friend_start.sh --config configs/friend_server.local.env' >&2; exit 2
fi
source "$2"
PLAN=configs/batch_plan.yaml
: "${CXR_DATA_ROOT:?Set authorized original ZIP path}"
: "${CXR_PROTOCOL_DIR:?Set separately authorized fixed protocol metadata directory}"
: "${CXR_OUTPUT_ROOT:?Set authorized output base directory}"
: "${CXR_PRETRAINED_ROOT:?Set authorized torch cache root}"
: "${CXR_BACKUP_ROOT:?Set authorized backup directory}"
: "${CXR_GPU_HOURS:?Set friend-approved maximum GPU hours}"
: "${CXR_GH_GPU_HOURS:?Set separately approved G/H GPU-hour subbudget for all 26 tasks}"
: "${CXR_GPUS:?Set explicit friend-authorized GPU IDs}"
: "${CXR_MAX_CONCURRENT:?Set authorized maximum concurrent tasks}"
: "${CXR_PREFLIGHT_VERSION:?Set the same version used by REMOTE_PREFLIGHT}"
[[ "$CXR_PREFLIGHT_VERSION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe preflight version' >&2; exit 2; }
GPUS="$CXR_GPUS"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run scripts/friend_remote_preflight.sh first; its setup creates .venv.' >&2; exit 2
fi
source .venv/bin/activate
python scripts/protocol_assets.py --config "$PLAN" --action import --directory "$CXR_PROTOCOL_DIR"
python scripts/batch_run.py --config "$PLAN" --action validate
ROOT="$CXR_OUTPUT_ROOT/phase17_batch"
RECEIPT_DIR="$CXR_OUTPUT_ROOT/phase17_batch_preflight_$CXR_PREFLIGHT_VERSION"
if [[ ! -f "$RECEIPT_DIR/receipt.json" ]]; then
  echo 'REMOTE_PREFLIGHT receipt missing; run scripts/friend_remote_preflight.sh first.' >&2; exit 2
fi
python scripts/batch_cost.py --config "$PLAN" --directory "$RECEIPT_DIR"
python scripts/batch_run.py --config "$PLAN" --action run-all --root "$ROOT" --gpus "$GPUS" \
  --max-concurrent "$CXR_MAX_CONCURRENT" --gpu-hours "$CXR_GPU_HOURS" \
  --gh-gpu-hours "$CXR_GH_GPU_HOURS" --receipt "$RECEIPT_DIR/receipt.json"
echo "Direct result folders: $ROOT"
