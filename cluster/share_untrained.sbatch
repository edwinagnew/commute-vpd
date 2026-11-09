#!/bin/bash
# E136 (cell P2.8): the share read on each saved state three ways (trained decomposition, each matrix's SVD, that SVD rotated). Plan:
# reports/plans/2026-11-09_plan_e136-share-untrained-decompositions.md; rule: experiments/share_untrained/results/predictions.json, pairs: results/inventory.json
# (run.py refuses unless both are committed unchanged).
#   cd ~/commute-agents/<role> && sbatch --comment=E136 cluster/share_untrained.sbatch
# Stage 0 (controls, alone; exit 1 on a miss) writes results/stage0.json; stage 1 (every state, resumes) writes results/stage1.json. The runner waits on `STAGE n DONE`.
# Inputs on NFS (gitignored): the weight changes and windows named in inventory.json, the model in artifacts/vpd/, E53's cached Pile val rows in artifacts/sets.
# Expected on a debug L40 (estimates; E60's 18 states read in ~90 s, job 46258): stage 0 ~6 min, stage 1 ~20 min; plan limit 110 min projected.
# Memory: two of E53's gate sets (500 windows x 127 tokens) are held during its route_split, ~20 GB on the host, so 40G.
#SBATCH --job-name=commute-share-untrained
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E136

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/share_untrained/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/share_untrained/run.py
