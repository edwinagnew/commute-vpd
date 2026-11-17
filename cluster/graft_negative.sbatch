#!/bin/bash
# E201 graft_negative. Usage: sbatch --comment=E201 cluster/graft_negative.sbatch
#SBATCH --job-name=commute-graft_negative
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/graft_negative/results/predictions.json || { echo "no predictions"; exit 1; }
( .venv/bin/python experiments/graft_negative/run.py 2>&1 | tee -a experiments/graft_negative/results/run.log; exit ${PIPESTATUS[0]} )
rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/graft_negative/results/ ==="
