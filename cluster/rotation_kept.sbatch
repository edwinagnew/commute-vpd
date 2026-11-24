#!/bin/bash
# E286 rotation_kept (P1.60). Usage: sbatch --comment='E286 stage: built' cluster/rotation_kept.sbatch <stage> <tag> [--shard I N]
# stage: control | built | e62 | head24 | trained | panel_nn | panel_vpd | gauge | summary (gauge skips control). Each job runs the control stage
# first (skipped once results/stage0_<tag>.json says done). panel_vpd goes to an H100 pod after one state is rehearsed here.
#SBATCH --job-name=commute-rotation-kept
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/rotation_kept/results/predictions.json || { echo "no predictions"; exit 1; }
STAGE="$1"; TAG="$2"; shift 2
LOG=experiments/rotation_kept/results/run_${STAGE}_${TAG}.log
( .venv/bin/python experiments/rotation_kept/run.py --stage "$STAGE" --tag "$TAG" "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/rotation_kept/results ===" ) 2>&1 | tee -a "$LOG"
