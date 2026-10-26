#!/usr/bin/env bash
# Submit with the FRIEND'S authorized partition/account/GPU/time limits via sbatch CLI.
# No resource request is embedded: do not allocate resources the friend has not approved.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this wrapper inside an authorized Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose the allocated GPU devices}"
export CXR_GPUS="$CUDA_VISIBLE_DEVICES"
bash scripts/friend_start.sh "$@"
