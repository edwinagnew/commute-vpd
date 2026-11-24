#!/bin/bash
# P2.67 (E284), the 67M: stage 0 alone first, then the items in shards. Usage, from the clone root:
#   PRED_SHA=$(git show HEAD:experiments/adl_published/results/predictions.json | sha256sum | cut -d' ' -f1)
#   PRED_SHA=$PRED_SHA sbatch --comment='E284 stage0' --export=ALL cluster/adl_published.sbatch stage0
#   PRED_SHA=$PRED_SHA sbatch --comment='E284 items 0/2' --export=ALL cluster/adl_published.sbatch items 0/2   # and 1/2
#SBATCH --job-name=commute-adl-pub
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS $*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
OUT=experiments/adl_published/results
( .venv/bin/python experiments/adl_published/run.py "$@"; echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/$OUT ===" ) 2>&1 | tee -a "$OUT/run_${1}_$(echo ${2:-0/1} | tr / -).log"
