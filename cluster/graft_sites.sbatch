#!/bin/bash
# E206 graft_sites (cell P1.28). Usage: sbatch --comment=E206 cluster/graft_sites.sbatch
#SBATCH --job-name=commute-graft_sites
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/graft_sites/results/predictions.json || { echo "no predictions"; exit 1; }
( .venv/bin/python experiments/graft_sites/run.py 2>&1 | tee -a experiments/graft_sites/results/run.log; exit ${PIPESTATUS[0]} )
rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/graft_sites/results/ ==="
