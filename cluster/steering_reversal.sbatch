#!/bin/bash
# E146 (cell P2.14): why steering reads the side built to be new higher on half-known vs cipher and rank-1 vs cipher.
# Plan: reports/plans/2026-11-10_plan_steering-reversal.md. Rule: experiments/steering_reversal/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E146 cluster/steering_reversal.sbatch
# Stage 0 (controls, alone) then stage 1 (28 saved updates, then the score); each prints "STAGE n DONE" after its results
# file. A filed stop prints "STOP: ..." and exits 1; stage 1 stops itself if the projected job time passes 25 min.
# Inputs on NFS, read only: E143's experiments/holes_fill/results/stage1.json, the updates and windows named in
# experiments/share_untrained/results/inventory.json for e54/elicit, e54/cipher, c3/rank1, c3/cipher, the model in
# artifacts/vpd/. Writes only experiments/steering_reversal/results/stage{0,1}.json; nothing large is saved.
# Expected on an L40: about 12 min (model load 2; stage 0 1; stage 1 about 8: 28 updates, 30 grid reads and up to 64
# random reads per update on 75 to 500 read rows; E143 read 50 updates with more passes in 10 min). Host memory: one
# update (113 MB) at a time, as E143 (24G, ran).
#SBATCH --job-name=commute-steering-reversal
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/steering_reversal/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/steering_reversal/run.py
