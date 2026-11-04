#!/bin/bash
# Upload E80's small inputs (A's offsets, the cut, W0's g statistics, ~440 MB) to the W&B artifact qwen3-1.7b-e80-inputs (experiments/qwen_d1_adversary/upload.py).
# A CPU batch job: the dev node's uplink is ~80 KB/s. Submit from the clone root, with WANDB_API_KEY in the submitting shell (sbatch exports it):
#   sbatch cluster/qwen_d1_adversary_upload.sbatch
# The log ends with the artifact's qualified name, minutes and the sha256 of each file. Expected version v0 (RUN_BRIEF.md says what to do if not).
#SBATCH --job-name=commute-qwen_d1_adversary_upload
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
.venv/bin/python experiments/qwen_d1_adversary/upload.py upload
