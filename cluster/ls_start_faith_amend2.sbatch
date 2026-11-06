#!/bin/bash
# E127 Amendment 2: each lowercase copy free, and its own change S_c placed in Delta on W0 (free adversary, no hold).
# Plan reports/plans/2026-11-06_plan_e127-ls-start-faithfulness.md -> "Amendment 2".
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   ~/commute-agents/fleet/resubmit --code 3 cluster/ls_start_faith_amend2.sbatch   # ~17 min on an L40: one job; exit 3 = more to do
# Before: experiments/ls_start_faith/results/predictions_amend2.json committed. Inputs on NFS:
# teacher_training/results/large/amend8/amend8c/uv0.pt, damage_crossed/results/large/{tt_c3,tt_recipe}.pt,
# broad_fragility/results/large/a2_tt_lora_s0.pt, transport_test/results/windows.json,
# transport_test/results/large/{alive,windows}.pt, artifacts/sets/val_0_256.npy (and what run.py's import loads).
# Writes experiments/ls_start_faith/results/ls_start_faith_amend2.json and results/large/amend2.log. A rerun continues from the file.
# Exit 1: a stop; the reason is `.stopped` in the file (move it aside to start again).
#SBATCH --job-name=commute-e127a2
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
[ $# -eq 0 ] || { echo "ls_start_faith_amend2.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/ls_start_faith/results/large
[ -d "$LOGS" ] || { echo "$LOGS missing: run ~/commute-agents/fleet/link-data <role> first" >&2; exit 1; }
rc=0
.venv/bin/python experiments/ls_start_faith/amend2.py 2>&1 | tee -a "$LOGS/amend2.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/ls_start_faith/results/ls_start_faith_amend2.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
