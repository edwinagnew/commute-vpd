#!/bin/bash
# E133 (cell P2.3): the original model's prior for the trained answer against the re-gating share, over every saved map on
# the 67M. Plan: reports/plans/2026-11-09_plan_e133-prior-vs-share.md. Rules: experiments/prior_vs_share/results/
# predictions.json, inventory.json (run.py refuses unless both are committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/prior_vs_share.sbatch
# Stage 0 (the control, alone) reproduces two saved priors; exits 1 on a miss. Stage 1 fills the priors with one forward
# per map missing one. Stage 2 reads the saved shares, computes the numbers and writes the scored file.
# Inputs on NFS (gitignored): windows under experiments/{regating_table,keydependent_elicit,open_pair,case_label,
# fresh_reuse_pair}/results/large/, the target model in artifacts/vpd/; those runs' JSON records (committed).
# Writes experiments/prior_vs_share/results/{stage0,stage1,stage2}.json only; commits nothing.
# Expected: under 5 minutes (one model load, about 14 small forwards, the bootstrap and permutations; E60's 18 states took
# ~90 s, job 46258, and E98's 56 with many forwards each ~4 min on an L40, TRIED.md rows E60, E98). The limit leaves room
# for a slow model load.
#SBATCH --job-name=commute-prior-vs-share
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
test -f experiments/prior_vs_share/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/prior_vs_share/run.py
