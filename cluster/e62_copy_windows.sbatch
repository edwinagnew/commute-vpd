#!/bin/bash
# E281 e62_copy_windows (P1.57). Usage: sbatch --comment="E281 read" cluster/e62_copy_windows.sbatch <stage> <tag> [--shard I N]
# stage: 0 | read. Each job runs stage 0 first (skipped once results/stage0_<tag>.json says done).
#SBATCH --job-name=commute-e62-copy-windows
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
test -f experiments/e62_copy_windows/results/predictions.json || { echo "no predictions"; exit 1; }
STAGE="$1"; TAG="$2"; shift 2
LOG=experiments/e62_copy_windows/results/run_${STAGE}_${TAG}.log
( .venv/bin/python experiments/e62_copy_windows/run.py --stage "$STAGE" --tag "$TAG" "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/e62_copy_windows/results ===" ) 2>&1 | tee -a "$LOG"
