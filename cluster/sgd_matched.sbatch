#!/bin/bash
# SGD matched: plain SGD vs Adam vs SGD+momentum at a matched size of change (reports/2026-10-21_plan_sgd-matched.md).
#   cd /mnt/nw/home/e.agnew/commute
#   sbatch cluster/sgd_matched.sbatch run --smoke    # smoke, ~3 min
#   sbatch cluster/sgd_matched.sbatch run            # scored, ~10 min on an L40
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json).
#SBATCH --job-name=commute-sgd-matched
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/sgd_matched/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/sgd_matched/run.py "$@"
