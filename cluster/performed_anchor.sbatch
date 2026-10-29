#!/bin/bash
# E53 performed anchor, stage A0: W0 only, no fine-tune (~2-5 min). Plan: reports/plans/2026-10-29_plan_performed-anchor.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/performed_anchor.sbatch a0
# Before: experiments/performed_anchor/results/predictions_a0.json filed and committed. Inputs: the VPD artifacts, Pile
# rows artifacts/sets/{val_0_256,train_0_4096}.npy (cached). Writes experiments/performed_anchor/results/a0.json and
# results/large/a0.log. Exit 0 = GO, 1 = STOP (the fine-tunes don't start), 2 = no positions (error).
#SBATCH --job-name=commute-e53-a0
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
[ "${1:-}" = a0 ] || { echo "usage: sbatch cluster/performed_anchor.sbatch a0" >&2; exit 2; }
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The shared .venv's editable install points at whichever clone last ran `uv sync`: put this clone's src first (as
# restore_anchors.sbatch). TEMPORARY: remove once the shared venv no longer pins one clone.
export PYTHONPATH="$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
test -f experiments/performed_anchor/results/predictions_a0.json || { echo "file the predictions first" >&2; exit 1; }
mkdir -p experiments/performed_anchor/results/large
rc=0
.venv/bin/python experiments/performed_anchor/a0.py 2>&1 | tee experiments/performed_anchor/results/large/a0.log || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/performed_anchor/results/a0.json ==="
exit $rc
