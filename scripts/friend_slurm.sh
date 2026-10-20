#!/usr/bin/env bash
# Submit with the FRIEND'S authorized partition/account/GPU/time limits via sbatch CLI.
# No resource request is embedded: do not allocate resources the friend has not approved.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this wrapper inside an authorized Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose the allocated GPU devices}"
# friend_run_all.sh reads the authorized pool from the local config. gpu_tokens
# also checks it is a subset of Slurm's CUDA_VISIBLE_DEVICES allocation.
bash scripts/friend_run_all.sh "$@"
