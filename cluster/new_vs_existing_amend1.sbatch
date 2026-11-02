#!/bin/bash
# E12 Amendment 1: the route split on teacher-trained states. Plan reports/plans/2026-11-02_plan_e12-teacher-route-split.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks, not a worktree
#   sbatch cluster/new_vs_existing_amend1.sbatch    # ~25 min on an L40 (12 students at 105 s, reads and load ~4 min)
# Before: results/predictions_amend1.json filed. Inputs on NFS: regating_table/results/large/{ident32,ident32b,rand32,rand32b}_s{0,1}.pt
# and windows_*.pt (sha256 in stage0a.json), teacher_training/results/large/amend8/n1/0.pt, transport_test/results/large/alive.pt.
# Writes results/new_vs_existing_amend1.json and results/large/amend1/ (12 files of 60 MB: 10 students' and 2 of C1's offsets).
# debug allows 120 min, so the 40 fit; a rerun on the same card skips every state with a record and an offsets file.
# Exit 1: a stop; the reason is `.stopped` in the file.
#SBATCH --job-name=commute-e12-amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 0 ] || { echo "new_vs_existing_amend1.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/new_vs_existing/results/large/amend1
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/new_vs_existing/amend1.py 2>&1 | tee -a "$LOGS/run_amend1.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/new_vs_existing/results/new_vs_existing_amend1.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
