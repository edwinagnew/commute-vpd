#!/bin/bash
# E121 Amendment 1: E34's adversarial check on the ladder's R3 and R4 states. Plan reports/plans/2026-11-06_plan_e121-copy-gap-ladder.md
# -> "Amendment 1".
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   ~/commute-agents/fleet/resubmit --code 3 cluster/copy_gap_ladder_amend1.sbatch   # ~27 min on an L40; exit 3 = more to do
# Before: experiments/copy_gap_ladder/results/predictions_amend1.json committed. Inputs on NFS: copy_gap_ladder/results/copy_gap_ladder.json,
# copy_gap_ladder/results/large/{F0,R3_o0,R3_o1,R3_o2,R4_start,R4_o1,R4_o2}.pt, teacher_training/results/large/amend8/amend8c/{uv0,real_teacher_dW}.pt,
# transport_test/results/windows.json, transport_test/results/large/{alive,windows}.pt, artifacts/sets/val_0_256.npy.
# Writes results/copy_gap_ladder_amend1.json, results/large/amend1/<state>.pt (bf16 offsets, ~60 MB each) and results/large/amend1/run.log.
# A rerun continues from the file. Exit 1: a stop; the reason is `.stopped` in the file (move it aside to start again).
#SBATCH --job-name=commute-e121a1
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
[ $# -eq 0 ] || { echo "copy_gap_ladder_amend1.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/copy_gap_ladder/results/large/amend1
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/copy_gap_ladder/amend1.py 2>&1 | tee -a "$LOGS/run.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/copy_gap_ladder/results/copy_gap_ladder_amend1.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
