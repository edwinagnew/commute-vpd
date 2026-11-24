#!/bin/bash
# E___ trained_reuse_control (cell P1.4 run 2). Usage: sbatch --comment='E___ <stage>' cluster/trained_reuse_control.sbatch <stage> <tag> [--shard I N]
#   train (--shard I 3, I = 0..2: one new_few and one x2 seed each, ~75 min), check (one job, ~31 min; R0c at neurons),
#   neurons (--shard I 3 after check passed, ~56 min each). vpd goes to one H100 pod (pod-runner), not here.
#   Each job runs its checks first and stops on a miss.
#SBATCH --job-name=commute-trained-reuse
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/trained_reuse_control/results/predictions.json || { echo "no predictions"; exit 1; }
STAGE="$1"; TAG="$2"; shift 2
LOG=experiments/trained_reuse_control/results/run_${STAGE}_${TAG}.log
( .venv/bin/python experiments/trained_reuse_control/run.py --stage "$STAGE" --tag "$TAG" "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/trained_reuse_control/results ===" ) 2>&1 | tee -a "$LOG"
