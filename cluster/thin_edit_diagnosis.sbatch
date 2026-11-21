#!/bin/bash
# E247 (cell P1.47): why the thin new edit reads kept under wording A at VPD, $0 on the cluster's debug QoS. One job:
# the do-nothing first and alone (stops on a miss), then E242's seven read items with E242's saved stood-in sets
# (experiments/thin_edit_diagnosis/run.py docstring). Plan reports/plans/2026-11-21_plan_thin-edit-diagnosis.md.
#   sbatch --comment=E247 cluster/thin_edit_diagnosis.sbatch
# PRED_SHA below is git show HEAD:experiments/thin_edit_diagnosis/results/predictions.json | sha256sum.
# --mem: E242's jobs peaked at 6.1 GB host RSS on the same items and code (cluster/logs/commute-wording-a-627*.out).
# Afterwards: ~/commute-agents/fleet/cpu-run python3 experiments/thin_edit_diagnosis/plot.py
#SBATCH --job-name=commute-thin-edit
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=9cf6b587a76fc297aa00b8d5f759fe86df8e7b1a5dfb790ba65472facf96b7e1 .venv/bin/python experiments/thin_edit_diagnosis/run.py ) 2>&1 \
    | tee -a "experiments/thin_edit_diagnosis/results/run.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E247 (exit $rc) — results in $COMMUTE_REPO/experiments/thin_edit_diagnosis/results/ ==="
exit $rc
