#!/bin/bash
# E84 (qwen_tofu_forms): the pod's files from W&B into results/large, each checked against the committed results/SHA256SUMS. A
# batch job, not fleet/cpu-run (a long transfer must outlive the session). About 68 GB, ~30 min. Submit from the clone root;
# needs WANDB_API_KEY in the submitting shell.
#SBATCH --job-name=commute-qwen_tofu_forms_fetch
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
.venv/bin/python experiments/qwen_tofu_forms/transfer.py down "$@"
