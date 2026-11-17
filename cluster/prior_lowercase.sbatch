#!/bin/bash
# E145 (cell P2.13): the original model's prior for the trained answer on E53's lowercase, cipher and broad cipher windows,
# the empty prior cell of the method table's lowercase-against-ciphers row. Plan: reports/plans/2026-11-10_plan_prior-lowercase.md.
# Rules: experiments/prior_lowercase/results/predictions.json (run.py refuses unless committed unchanged).
#   cd ~/commute-agents/<role> && sbatch --comment='E145 prior on lowercase against ciphers' cluster/prior_lowercase.sbatch
# Stage 0 (alone): E133's forward re-reads e58_24_rank1's prior (5.6705e-3, 1e-4 relative), and the rebuilt E53 windows give
# E53's saved W0 log-odds per task (0.01 nats); a miss exits 1. Stage 1: per-task prior, interval over windows, gap, reading.
# Inputs: cached Pile val rows (the E53 windows are rebuilt from them, checked on the CPU worker 2026-11-10: 500 x 127 per
# task), experiments/open_pair/results/large/amend1/windows_rank1_trained.pt, the target model in artifacts/vpd/, and the
# committed JSON records named in predictions.json. Writes experiments/prior_lowercase/results/{stage0,stage1}.json only.
# Expected: under 5 minutes (model load 1-2 min, four full-batch forwards of at most 500 windows; E60 read these windows on
# the same path in ~90 s, job 46258).
#SBATCH --job-name=commute-prior-lowercase
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/prior_lowercase/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/prior_lowercase/run.py
