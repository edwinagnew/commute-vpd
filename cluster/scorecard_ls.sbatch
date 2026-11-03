#!/bin/bash
# E71 scorecard_ls. Plan: reports/plans/2026-11-03_plan_e71-scorecard-ls.md.
#   cd ~/commute-agents/<role>; sbatch cluster/scorecard_ls.sbatch
# Before: experiments/scorecard_ls/results/predictions.json filed and committed. Inputs on NFS: row_blocks' dW_{adam,lora}.pt,
# transport_test's alive.pt and windows.pt. Outputs: experiments/scorecard_ls/results/{reads.json,run.log}.
#SBATCH --job-name=commute-e71
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/scorecard_ls/run.py || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/scorecard_ls/results/reads.json ==="
exit $rc
