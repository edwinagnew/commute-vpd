#!/bin/bash
# Upload the converted Qwen3-1.7B decomposition (53.65 GB) to the W&B artifact qwen3-1.7b-p-e9fcffd5-torch, so a
# RunPod pod can fetch it (experiments/qwen_port/upload.py, src/commute/qwen_files.py). A batch job, not fleet/cpu-run:
# the first upload ran under cpu-run's srun and died with the session that started it (2026-10-27, job 38973).
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_upload.sbatch      # needs WANDB_API_KEY in the submitting shell (sbatch exports it)
# Output: the log ends with the artifact's qualified name, size, minutes and MB/s.
#SBATCH --job-name=commute-qwen_upload
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
.venv/bin/python -u experiments/qwen_port/upload.py "$COMMUTE_QWEN_DIR"
