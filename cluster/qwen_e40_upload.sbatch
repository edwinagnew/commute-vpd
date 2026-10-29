#!/bin/bash
# Upload the remade E40 δW (cluster/qwen_e40_regen.sbatch's output) to the W&B artifact qwen3-1.7b-e40-nowarmup-dW
# (src/commute/qwen_e40_files.py). A CPU batch job: the dev node's uplink is ~80 KB/s. Submit from the clone root,
# with WANDB_API_KEY in the submitting shell (sbatch exports it):
#   sbatch cluster/qwen_e40_upload.sbatch
# The log ends with the artifact's qualified name, minutes and the sha256 of each file.
#SBATCH --job-name=commute-qwen_e40_upload
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
.venv/bin/python -m commute.qwen_e40_files upload "$HOME/commute-data/results-large/qwen_sft/nowarmup_regen"
