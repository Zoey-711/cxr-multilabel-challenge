#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 2 || "$1" != "--config" ]]; then
  echo 'Usage: bash scripts/friend_start.sh --config configs/batch_plan.yaml' >&2; exit 2
fi
PLAN="$2"
: "${CXR_DATA_ROOT:?Set authorized original ZIP path}"
: "${CXR_PROTOCOL_BUNDLE:?Set separately authorized fixed protocol metadata archive}"
: "${CXR_OUTPUT_ROOT:?Set authorized output base directory}"
: "${CXR_PRETRAINED_ROOT:?Set authorized torch cache root}"
: "${CXR_BACKUP_ROOT:?Set authorized backup directory}"
: "${CXR_GPU_HOURS:?Set friend-approved maximum GPU hours}"
GPUS="${CXR_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! -x .venv/bin/python ]]; then
  bash scripts/cloud_install.sh --config configs/batch17/d17_fold1.yaml
fi
source .venv/bin/activate
python scripts/protocol_assets.py --config "$PLAN" --action import --file "$CXR_PROTOCOL_BUNDLE"
python scripts/batch_run.py --config "$PLAN" --action validate
ROOT="$CXR_OUTPUT_ROOT/phase17_batch"
RECEIPT_DIR="$CXR_OUTPUT_ROOT/phase17_batch_preflight"
if [[ ! -f "$RECEIPT_DIR/receipt.json" ]]; then
  python scripts/batch_preflight.py --config "$PLAN" --output "$RECEIPT_DIR" --gpus "$GPUS" --gpu-smoke --benchmark --budget-hours "$CXR_GPU_HOURS"
fi
python scripts/batch_cost.py --config "$PLAN" --directory "$RECEIPT_DIR"
ARGS=()
if [[ -f "$ROOT/queue_state.json" ]]; then ARGS+=(--resume); fi
if [[ "${CXR_ENABLE_EXTENDED:-0}" == 1 ]]; then
  echo 'Extended tasks require separate user approval; use explicit batch_run --allow-extended after approval.' >&2; exit 2
fi
python scripts/batch_run.py --config "$PLAN" --action run --root "$ROOT" --gpus "$GPUS" \
  --gpu-hours "$CXR_GPU_HOURS" --receipt "$RECEIPT_DIR/receipt.json" "${ARGS[@]}"
echo "Result packages: $ROOT/delivery/ (see latest.json and COMPLETE.txt)"
