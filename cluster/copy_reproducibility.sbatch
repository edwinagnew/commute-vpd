#!/bin/bash
# E126: E119's emoticon Adam Pile-text copy and the three re-switching Pile-text copies at batch orders 1-2, on the L40.
# Plan reports/plans/2026-11-06_plan_e126-copy-reproducibility.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   sbatch cluster/copy_reproducibility.sbatch      # ~26 min of compute (E119's L40 log); exit 3 = more to do:
#   ~/commute-agents/fleet/resubmit --code 3 cluster/copy_reproducibility.sbatch   # resubmits until it ends otherwise
# Before: experiments/copy_reproducibility/results/predictions.json committed. Inputs on NFS: transport_test/results/large/
# {alive,windows}.pt, transport_test/results/windows.json, task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt,
# teacher_route_split/results/teacher_route_split_*.json, artifacts/sets/train_32768_8192.npy.
# Writes results/copy_reproducibility.json and results/large/<task>/<opt>_B_o<k>.pt (~60 MB each, nine) and run.log.
# A rerun skips recorded students. Exit 1: a stop; the reason is `.stopped` in the file (move it aside to start again).
#SBATCH --job-name=commute-e126
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
[ $# -eq 0 ] || { echo "copy_reproducibility.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/copy_reproducibility/results/large
[ -d "$LOGS" ] || { echo "$LOGS missing: run ~/commute-agents/fleet/link-data <role> first" >&2; exit 1; }
rc=0
.venv/bin/python experiments/copy_reproducibility/run.py 2>&1 | tee -a "$LOGS/run.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/copy_reproducibility/results/copy_reproducibility.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
