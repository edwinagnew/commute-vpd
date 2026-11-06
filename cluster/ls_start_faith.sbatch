#!/bin/bash
# E127 ls_start_faith: why the least-squares start reads worse than do-nothing under E34's adversarial check.
# Plan reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   ~/commute-agents/fleet/resubmit --code 3 cluster/ls_start_faith.sbatch   # ~30 min on an L40: two jobs; exit 3 = more to do
# Before: experiments/ls_start_faith/results/predictions.json committed. Inputs on NFS: copy_gap_ladder/results/large/{F0.pt,
# amend1/R3_o0.pt, amend1/R4_o1.pt}, teacher_training/results/large/amend8/amend8c/{uv0,real_teacher_dW}.pt,
# damage_crossed/results/large/{ft_adam300,tt_c3_start,tt_c3,tt_recipe}.pt, transport_test/results/windows.json,
# transport_test/results/large/{alive,windows}.pt, artifacts/sets/val_0_256.npy.
# Writes experiments/ls_start_faith/results/ls_start_faith.json and results/large/run.log. A rerun continues from the file.
# Exit 1: a stop; the reason is `.stopped` in the file (move it aside to start again).
#SBATCH --job-name=commute-e127
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
[ $# -eq 0 ] || { echo "ls_start_faith.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/ls_start_faith/results/large
[ -d "$LOGS" ] || { echo "$LOGS missing: run ~/commute-agents/fleet/link-data <role> first" >&2; exit 1; }
rc=0
.venv/bin/python experiments/ls_start_faith/run.py 2>&1 | tee -a "$LOGS/run.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/ls_start_faith/results/ls_start_faith.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
