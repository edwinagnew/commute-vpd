#!/bin/bash
# E290 ia3_control (cell P1.61). Usage: sbatch --comment='E290 <stage>' cluster/ia3_control.sbatch <stage> <tag> [--shard I N]
#   stage train (one job), neurons (--shard I 8), vpd (--shard I 10); each job runs its stage 0 first and stops on a miss.
#SBATCH --job-name=commute-ia3-control
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/ia3_control/results/predictions.json || { echo "no predictions"; exit 1; }
STAGE="$1"; TAG="$2"; shift 2
LOG=experiments/ia3_control/results/run_${STAGE}_${TAG}.log
( .venv/bin/python experiments/ia3_control/run.py --stage "$STAGE" --tag "$TAG" "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/ia3_control/results ===" ) 2>&1 | tee -a "$LOG"
