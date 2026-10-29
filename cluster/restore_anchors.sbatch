#!/bin/bash
# E50, the restore share on E12's known-kind anchors. Plan: reports/plans/2026-10-29_plan_restore-anchors.md ("Filed (E50)").
#   cd ~/commute-agents/<role>
#   sbatch cluster/restore_anchors.sbatch   # controls first on all eight states (exit 1 on a miss), then the grid (~6-10 min)
#   sbatch cluster/restore_anchors.sbatch amend1   # Amendment 1, power checks (plan section "Amendment 1"; ~3-6 min):
#     also reads experiments/restore_anchors/results/{predictions_amend1.json, restore_anchors.json} and
#     experiments/regating_table/results/{stage0b.json, amend1.json, amend3.json}; writes results/amend1.json, large/amend1.log
# Before that: predictions filed (experiments/restore_anchors/results/predictions.json). Inputs on NFS (gitignored):
# experiments/regating_table/results/large/{windows_<table>.pt, <table>_s{0,1}.pt} (E46 stage 0a, checked by the sha256
# in experiments/regating_table/results/stage0a.json), experiments/transport_test/results/large/alive.pt, Pile rows
# artifacts/sets/val_0_2000.npy (streamed from Hugging Face on first use). Writes
# experiments/restore_anchors/results/restore_anchors.json and results/large/run.log.
#SBATCH --job-name=commute-restore-anchors
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The shared .venv's editable install points at whichever clone last ran `uv sync`: put this clone's src first (as
# regating_table.sbatch). TEMPORARY: remove once the shared venv no longer pins one clone.
export PYTHONPATH="$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
test -f experiments/restore_anchors/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
case "${1:-}" in
  "") .venv/bin/python experiments/restore_anchors/run.py ;;
  amend1) .venv/bin/python experiments/restore_anchors/amend1.py ;;  # refuses without results/predictions_amend1.json
  *) echo "usage: sbatch cluster/restore_anchors.sbatch [amend1]"; exit 2 ;;
esac
