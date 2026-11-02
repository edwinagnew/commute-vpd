#!/bin/bash
# E67: the weight-side read. No training. Γ's change at every position of 128 Pile rows, and the overlap shares S_X, on saved fine-tunes.
# Plan: reports/plans/2026-11-02_plan_e67-weight-side-read.md. Rules: experiments/weight_side_read/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/weight_side_read.sbatch
# Exit 1 at: any control (stage a: size, zero, ones), a shares' sum off 1 at any state. A verdict against the filed lean is a LOUD line, no stop.
# Inputs (read): experiments/regating_table/results/{stage0a.json,large/{ident32,rand32}_s{0,1}.pt,large/code_step{10,100}.pt} (sha256 checked),
# experiments/new_vs_existing/results/{new_vs_existing_amend1.json,large/amend1/*.pt}, experiments/transport_test/results/large/alive.pt, the VPD artifacts, Pile val rows.
# Writes experiments/weight_side_read/results/run.json after each stage (refuses if it exists) and results/large/run.out. Deletes nothing, commits nothing.
# results/large is a symlink into commute-data; a new experiment has none until link-data runs, so this job runs it (idempotent, touches links only).
# Time: expected 10-20 min (plan, "Cost"), a guess until the first state's `seconds` prints: fp32 Γ twice per chunk, 8 chunks, 7 states. The 30-min limit also catches a hang.
#SBATCH --job-name=commute-e67-weight-side
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
large=experiments/weight_side_read/results/large
[ -e "$large" ] || "$HOME/commute-agents/fleet/link-data" "$COMMUTE_REPO"
if [ -e "$large/run.out" ]; then echo "$large/run.out exists: move it aside, this run deletes nothing"; exit 1; fi
rc=0
.venv/bin/python experiments/weight_side_read/run.py "$@" 2>&1 | tee "$large/run.out" || rc=$?
echo "=== FINISHED E67 (exit $rc) — results in $COMMUTE_REPO/experiments/weight_side_read/results/run.json ==="
exit $rc
