#!/bin/bash
# E161 (cell P2.6), share_stability: the every-position share on the saved code-text states over five 64-row batches, and one
# Gaussian update of the step-100 state's per-matrix Frobenius norm. No training.
# Plan: reports/plans/2026-11-11_plan_p2-6-share-stability.md. Rules: experiments/share_stability/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch --comment='E161 P2.6 share stability' cluster/share_stability.sbatch
# Exit 1 at stage 0 (zero update not exactly W0, rows or states not E46's, batch 0 not reproducing E46's cells within
# 2e-4 or r within 0.01). `STAGE n DONE` after each stage's write to results/run.json.
# Inputs (read): experiments/regating_table/results/{stage0a.json,large/code_held.pt,large/code_step{10,100}.pt} (sha256
# checked), experiments/regating_spread/results/regating_spread.json, the VPD artifacts, Pile train and val rows.
# Time: E46 read both code states on these 64 rows in 6 s and selected the rows in under 26 s on an L40
# (regating_table/results/large/full.log, "[time] 0b code", "[time] 0a e36/code"); this does about 21 such 64-row reads:
# expected 2-5 min with model load. Limit 30 min, no resume.
#SBATCH --job-name=commute-share-stability
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/share_stability/run.py "$@" || rc=$?
echo "=== FINISHED share_stability (exit $rc) — results in $COMMUTE_REPO/experiments/share_stability/results/run.json ==="
exit $rc
