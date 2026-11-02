#!/bin/bash
# E12 Amendment 3: does rand32b's fall survive reading Gamma and the masked logits in fp32? Plan reports/plans/2026-11-02_plan_e12-amend3-fp32-read.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks, not a worktree
#   sbatch cluster/new_vs_existing_amend3.sbatch    # ~2 min on an L40 (model load 23 s, 12 pairs of reads); read only, no training
# Before: results/predictions_amend3.json and results/new_vs_existing_amend2.json present. Inputs on NFS:
# new_vs_existing/results/large/amend2/*.pt, regating_table/results/large/{rand32,rand32b}_s{0,1}.pt and windows_*.pt, transport_test/results/large/alive.pt.
# Writes results/amend3.json and results/large/amend3/run_amend3.log. A rerun refuses while the results file exists: move it aside.
# Exit 1: a stop; the reason is `.stopped` in the file.
#SBATCH --job-name=commute-e12-amend3
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 0 ] || { echo "new_vs_existing_amend3.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/new_vs_existing/results/large/amend3
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/new_vs_existing/amend3.py 2>&1 | tee -a "$LOGS/run_amend3.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/new_vs_existing/results/amend3.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
