#!/bin/bash
# E268 kstar_needed. Usage: sbatch --comment='E268 stage: built' cluster/kstar_needed.sbatch <stage> <tag> [--shard I N]
#SBATCH --job-name=commute-kstar-needed
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/kstar_needed/results/predictions.json || { echo "no predictions"; exit 1; }
STAGE="$1"; TAG="$2"; shift 2
LOG=experiments/kstar_needed/results/run_${STAGE}_${TAG}.log
( .venv/bin/python experiments/kstar_needed/run.py --stage "$STAGE" --tag "$TAG" "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/kstar_needed/results ===" ) 2>&1 | tee -a "$LOG"
