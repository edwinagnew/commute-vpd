#!/bin/bash
# E113 (plan reports/plans/2026-11-06_plan_e113-case-patch.md): patch the case readout's coordinate at block 1 (trigger:
# E112's previous-token direction; word: the own-token direction) to the other case's Pile mean on E112's four saved
# states. Q0 on W0 first and alone (exit 1 if block 2's probe reads the patched case on < 0.70); the random-direction
# control on every state next (exit 1 if a draw flips > 0.10); then Q1 and the verdict per state; beside numbers last.
# Rules: experiments/case_patch/results/predictions.json and experiments/case_label/results/predictions.json (run.py
# refuses unless both are committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/case_patch.sbatch
# Exit 0 done; 1 Q0 or the control missed (no state read), or a saved state's sha256 differs (assert).
# Inputs: experiments/case_label/results/large/{task,probes}.pt and dW_{real,twin}_s{0,1}.pt, E112's stage{0,1,2}.json,
# the target model in artifacts/vpd/.
# Writes experiments/case_patch/results/run.json (after every step) and run.out (this log, appended). No large files.
# Expected time on a debug L40 (estimate): about 4 min (model load ~1; residuals of 9,600 + 1,600 windows, ~60 passes of
# 1,024-9,600 windows of 41 tokens < 3). Memory: E112 Amendment 1, the same reads, peaked at 6.1 GB host; 16G.
#SBATCH --job-name=commute-e113
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/case_patch/run.py 2>&1 | tee -a experiments/case_patch/results/run.out || rc=$?
echo "=== FINISHED E113 (exit $rc) — results in $COMMUTE_REPO/experiments/case_patch/results/ ==="
exit $rc
