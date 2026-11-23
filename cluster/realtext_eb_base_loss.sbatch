#!/bin/bash
# E273 stage "base loss" (cell P2.64): M's all-position loss on E273's six real-text sides, experiments/realtext_eb/base_loss.py.
#   sbatch --comment="E273 stage: base loss" cluster/realtext_eb_base_loss.sbatch
#SBATCH --job-name=commute-realtext-base-loss
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
LOG=experiments/realtext_eb/results/run_base_loss.log
.venv/bin/python experiments/realtext_eb/base_loss.py 2>&1 | tee -a "$LOG"
CODE=${PIPESTATUS[0]}
echo "=== FINISHED (exit $CODE) — results in $COMMUTE_REPO/experiments/realtext_eb/results/base_loss.json ===" | tee -a "$LOG"
exit $CODE
