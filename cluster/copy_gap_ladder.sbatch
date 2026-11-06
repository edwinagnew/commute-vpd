#!/bin/bash
# E121: the copy-gap ladder. Plan reports/plans/2026-11-06_plan_e121-copy-gap-ladder.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   sbatch cluster/copy_gap_ladder.sbatch           # ~20 min on an L40 (filing's expected."time on the L40"); exit 3 = more to do:
#   ~/commute-agents/fleet/resubmit --code 3 cluster/copy_gap_ladder.sbatch   # resubmits until it ends otherwise
# Before: experiments/copy_gap_ladder/results/predictions.json committed. Inputs on NFS: teacher_training/results/amend10.json,
# teacher_training/results/large/amend8/amend8c/{real_teacher_dW,real_o0,real_o1,uv0}.pt, .../large/amend10/real_teacher_o{1,2}_dW.pt,
# transport_test/results/large/windows.pt, transport_test/results/windows.json, artifacts/sets/val_0_256.npy.
# Writes results/copy_gap_ladder.json and results/large/{F0,R3_o0,R3_o1,R4_start,R*_o1,R*_o2}.pt (~57 MB each, F0 113 MB) and run.log.
# A rerun continues from those files. Exit 1: a stop; the reason is `.stopped` in the file (move it aside to start again).
#SBATCH --job-name=commute-e121
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
[ $# -eq 0 ] || { echo "copy_gap_ladder.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/copy_gap_ladder/results/large
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/copy_gap_ladder/run.py 2>&1 | tee -a "$LOGS/run.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/copy_gap_ladder/results/copy_gap_ladder.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
