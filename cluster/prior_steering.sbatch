#!/bin/bash
# E149 (cell P2.15): steering at strength 1 on every built pair's saved states, then prior and steering together against the share.
# Plan: reports/plans/2026-11-10_plan_prior-steering.md. Rule: experiments/prior_steering/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E149 cluster/prior_steering.sbatch
# Stage 0 (controls, alone) then stage 1 (68 saved updates of the 9 built pairs, then the score); each prints "STAGE n DONE"
# after its results file. A filed stop prints "STOP: ..." and exits 1; stage 1 stops itself if the projected job time passes 25 min.
# Inputs on NFS, read only: final.json, E136's inventory and the updates and windows it names, E143's, E135's, E146's and E147's
# results, E62's windows and six updates, E147's twelve 96-key refits (experiments/refit_96key/results/large/dW_{rank1,cipher,
# copy,prior_cipher}_s{0,1,2}.pt, sha256 checked against E147's stage0.json), the model in artifacts/vpd/. E147's stage1.json
# must be in this clone (it is read for its saved grids and its filed steering calls). Writes only
# experiments/prior_steering/results/stage{0,1}.json; nothing large is saved.
# Expected on an L40: about 12 min (model load 2; E62's states and baselines 1; stage 0 2; stage 1 about 7: 68 updates, each
# vectors over its training windows, 6 selection reads on 512 rows and 2 read-row reads; E146 read 28 updates with about 90
# reads each in 8 min). Host memory: E62's six updates (113 MB each) plus one update at a time, as E143 and E146 (24G, ran).
#SBATCH --job-name=commute-prior-steering
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/prior_steering/results/predictions.json || { echo "file the predictions first"; exit 1; }
test -f experiments/refit_96key/results/stage1.json || { echo "E147's stage1.json is not in this clone"; exit 1; }
.venv/bin/python experiments/prior_steering/run.py
