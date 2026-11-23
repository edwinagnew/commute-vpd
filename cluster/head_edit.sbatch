#!/bin/bash
# E280 head_edit (P1.57). Usage: sbatch --comment='E280 read vpd' cluster/head_edit.sbatch <stage> <feat> <tag>
# stage: 0 | read; feat: vpd | neurons | both. Each job runs stage 0 first per featurization (skipped once
# results/stage0_<feat>_<tag>.json says done).
#SBATCH --job-name=commute-head-edit
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
test -f experiments/head_edit/results/predictions.json || { echo "no predictions"; exit 1; }
STAGE="$1"; FEAT="$2"; TAG="$3"
LOG=experiments/head_edit/results/run_${STAGE}_${FEAT}_${TAG}.log
( .venv/bin/python experiments/head_edit/run.py --stage "$STAGE" --feat "$FEAT" --tag "$TAG"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/head_edit/results ===" ) 2>&1 | tee -a "$LOG"
