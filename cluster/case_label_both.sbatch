#!/bin/bash
# E123: E112's case-label pair read by the decomposition side. Plan reports/plans/2026-11-06_plan_e123-case-label-both-readings.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   sbatch cluster/case_label_both.sbatch           # ~17 min on an L40 (predictions expected."time on the L40"); exit 3 = more to do:
#   ~/commute-agents/fleet/resubmit --code 3 cluster/case_label_both.sbatch   # resubmits until it ends otherwise
# Before: experiments/case_label_both/results/predictions.json committed. Inputs on NFS: case_label/results/large/{task.pt,
# dW_{real,twin}_s{0,1}.pt}, case_label/results/stage{0,1,2}.json, case_locator/results/stage0.json,
# transport_test/results/large/alive.pt, new_vs_existing/results/new_vs_existing_amend1.json, recruitment_test/results/amend2.json.
# Writes results/case_label_both.json, results/large/students/<state>.pt (bf16 offsets, ~60 MB each),
# results/large/shift_<state>.pt and run.log. A rerun continues: stage 0 once passed, each record once written.
# Exit 1: a stop; the reason is `stopped` in the file (move it aside to start again).
#SBATCH --job-name=commute-e123
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 0 ] || { echo "case_label_both.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/case_label_both/results/large
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/case_label_both/run.py 2>&1 | tee -a "$LOGS/run.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/case_label_both/results/case_label_both.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
