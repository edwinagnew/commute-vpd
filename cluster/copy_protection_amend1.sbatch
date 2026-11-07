#!/bin/bash
# E130 Amendment 1: c1 (and c2) placed, in Delta and c1 do-nothing on Pile val rows 256-511, E130 check.
# Plan reports/plans/2026-11-07_plan_e130-copy-protection.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   ~/commute-agents/fleet/resubmit --code 3 cluster/copy_protection_amend1.sbatch   # ~14 min on an L40: one job; exit 3 = more to do
# Before: experiments/copy_protection/results/predictions_amend1.json committed. Inputs on NFS:
# teacher_training/results/large/amend8/amend8c/uv0.pt, damage_crossed/results/damage_crossed.json,
# damage_crossed/results/large/{tt_c1,tt_c2,ft_c1}.pt, transport_test/results/windows.json,
# transport_test/results/large/{alive,windows}.pt, artifacts/sets/val_256_256.npy (and the decomposition harness.load reads).
# Writes experiments/copy_protection/results/copy_protection_amend1.json and results/large/amend1.log. A rerun continues from the file.
# Exit 1: a stop; the reason is `.stopped` in the file (move it aside to start again). Peak host RSS ~6-7 GB (E127 A2 6.1 GB).
#SBATCH --job-name=commute-e130a1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 0 ] || { echo "copy_protection_amend1.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/copy_protection/results/large
[ -d "$LOGS" ] || { echo "$LOGS missing: run ~/commute-agents/fleet/link-data <role> first" >&2; exit 1; }
rc=0
.venv/bin/python experiments/copy_protection/amend1.py 2>&1 | tee -a "$LOGS/amend1.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/copy_protection/results/copy_protection_amend1.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
