#!/bin/bash
# lora_matched: rank-8 LoRA on all 24 matrices vs the full-weight fine-tune, both lifted (design: section 3.2 of
# reports/2026-10-19_review_lora-failure-and-fixes.md). One job; the arguments go to run.py:
#   cd /mnt/nw/home/e.agnew/commute
#   sbatch --job-name=commute-lora_matched-smoke --time=00:20:00 cluster/lora_matched.sbatch all --smoke
#   sbatch cluster/lora_matched.sbatch all
# Before that: `cluster/sync.sh up` from the laptop, predictions filed (results/predictions.json).
# Expected: smoke ~5 min; scored ~20 min (7 Pile evaluations at ~2 min each, two LoRA trainings, one 10-step fine-tune).
#SBATCH --job-name=commute-lora_matched
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/lora_matched/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/lora_matched/run.py "$@"
