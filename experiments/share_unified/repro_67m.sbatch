#!/bin/bash
# Cell P2.90: the 67M paper re-gating shares re-read with the one share (experiments/share_unified/repro_67m.py).
# 48 states + the zero update, ~3 s each on an L40 (reading_refactor new_a share_e136: 4.9 s for two states), so a few
# minutes with the model load. Writes experiments/share_unified/results/repro_67m.jsonl, one line per state.
#   sbatch --comment='E### P2.90 repro_67m' experiments/share_unified/repro_67m.sbatch
#SBATCH --job-name=commute-p290-repro67m
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the checkout's root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS commit=$(git rev-parse HEAD)" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/share_unified/repro_67m.py
