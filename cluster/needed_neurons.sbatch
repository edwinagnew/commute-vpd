#!/bin/bash
# E277 needed_neurons. Usage: sbatch --comment='E277 stage: built' cluster/needed_neurons.sbatch <stage> <tag> [--shard I N]
#SBATCH --job-name=commute-needed-neurons
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
test -f experiments/needed_neurons/results/predictions.json || { echo "no predictions"; exit 1; }
STAGE="$1"; TAG="$2"; shift 2
LOG=experiments/needed_neurons/results/run_${STAGE}_${TAG}.log
( .venv/bin/python experiments/needed_neurons/run.py --stage "$STAGE" --tag "$TAG" "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/needed_neurons/results ===" ) 2>&1 | tee -a "$LOG"
