#!/bin/bash
# sgd_finetune: SGD fine-tune rate, then splits and SGD component training (reports/plans/2026-10-21_plan_sgd-finetune.md).
#   cd ~/commute-agents/<role>
#   sbatch cluster/sgd_finetune.sbatch rate     # ~5 min on an L40
#   sbatch cluster/sgd_finetune.sbatch split    # ~23 min on an L40; only if rate passed
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json).
#SBATCH --job-name=commute-sgd-finetune
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/sgd_finetune/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/sgd_finetune/run.py "$@"
