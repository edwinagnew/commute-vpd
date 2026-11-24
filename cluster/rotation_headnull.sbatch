#!/bin/bash
# P1.60 run 2 (rotation_kept run_headnull.py): copy head (2, 4) beside six size-matched random updates.
# Usage: sbatch --comment=E___ cluster/rotation_headnull.sbatch <tag> <I> <N>   (bases I, I+N, ... and their randoms)
# Both shards: ... h0 0 2 and ... h1 1 2. Each job runs stage 0 first (identity, do-nothing, E280's v2) and stops on a miss.
#SBATCH --job-name=commute-rotation-headnull
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
test -f experiments/rotation_kept/results/predictions.json || { echo "no predictions"; exit 1; }
TAG="$1"; I="$2"; N="$3"
mkdir -p experiments/rotation_kept/results/headnull
LOG=experiments/rotation_kept/results/headnull/run_${TAG}.log
( .venv/bin/python experiments/rotation_kept/run_headnull.py --tag "$TAG" --shard "$I" "$N"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/rotation_kept/results/headnull ===" ) 2>&1 | tee -a "$LOG"
