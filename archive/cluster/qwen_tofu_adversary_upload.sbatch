#!/bin/bash
# Upload E81's TOFU update (one fp32 δW, 5.64 GB) to the W&B artifact qwen3-1.7b-e81-tofu-dW (experiments/qwen_tofu_adversary/upload.py).
# A CPU batch job: W&B lets a pod fetch at 30-45 MB/s (a direct sync() from the dev node is ~10 MB/s, measured 2026-11-11). Submit from the clone root, with WANDB_API_KEY in the submitting shell (sbatch exports it):
#   sbatch cluster/qwen_tofu_adversary_upload.sbatch
# The log ends with the artifact's qualified name, minutes and the sha256. Expected version v0, which experiments/qwen_tofu_adversary/run.py fetches.
#SBATCH --job-name=commute-qwen_tofu_adversary_upload
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
.venv/bin/python experiments/qwen_tofu_adversary/upload.py upload
