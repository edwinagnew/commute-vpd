#!/bin/bash
# The re-gating side of the step-2 table (reports/plans/2026-10-30_plan_regating-ground-truth.md), saved states only, $0:
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_truth_table.sbatch   # stage 0 controls (exit 1 on a miss), S on E53's six states, the table
#                                                 # (~5-10 min): results/stage0.json, stage1.json, table.json
# Before: results/predictions.json filed and committed. Inputs (read only, sha256-checked): E46's
# experiments/regating_table/results/large/{ident32,ident32b,rand32,rand32b}_s{0,1}.pt and windows_ident32.pt, E53's
# experiments/performed_anchor/results/large/dW_{lowercase,cipher,broad_cipher}_s{0,1}_step300.pt, E54's
# experiments/keydependent_elicit/results/large/{elicit,cipher}_s{0,1}.pt and windows_elicit.pt, the VPD artifacts,
# Pile rows artifacts/sets/{val_0_256,train_0_4096}.npy, experiments/transport_test/results/large/alive.pt.
# Log: experiments/regating_truth_table/results/large/run.out. Exit 1 = a control missed or a state's block-0 shift.
#SBATCH --job-name=commute-truth-table
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# The shared .venv's editable install points at whichever clone last ran `uv sync`: put this clone's src first (as
# performed_anchor.sbatch). TEMPORARY: remove once the shared venv no longer pins one clone.
export PYTHONPATH="$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
test -f experiments/regating_truth_table/results/predictions.json || { echo "file the predictions first" >&2; exit 1; }
mkdir -p experiments/regating_truth_table/results/large
rc=0
.venv/bin/python experiments/regating_truth_table/run.py 2>&1 | tee experiments/regating_truth_table/results/large/run.out || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/regating_truth_table/results/ ==="
exit $rc
