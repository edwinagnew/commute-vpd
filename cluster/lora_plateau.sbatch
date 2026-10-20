#!/bin/bash
# lora_plateau: round 5's LoRA to 1000 steps, the inexpressible share against A's movement, Pile at steps 10, 100, 1000.
# Design check confirmed by the user 2026-10-19 (after reports/2026-10-19_analysis_lora-matched.md). One job; the arguments go to run.py:
#   cd /mnt/nw/home/e.agnew/commute
#   sbatch --job-name=commute-lora_plateau-smoke --time=00:20:00 cluster/lora_plateau.sbatch all --smoke
#   sbatch cluster/lora_plateau.sbatch all
# Before that: `cluster/sync.sh up` from the laptop, predictions filed (results/predictions.json).
# Expected: smoke ~5 min; scored ~30 min (9 Pile evaluations at ~2 min each, 1000 LoRA steps, 9 checkpoint measurements).
#SBATCH --job-name=commute-lora_plateau
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/lora_plateau/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/lora_plateau/run.py "$@"
