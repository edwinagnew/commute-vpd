#!/bin/bash
# P2.28 (E???), the 67M half: the nine built pairs' larger-side updates rescaled to the smaller side's size and read again
# (share, steering, top neurons, k_half, erank). Rules: experiments/size_matched/results/predictions.json (run.py refuses
# unless committed unchanged and numbered).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E??? cluster/size_matched.sbatch
# Stage 0 alone (sizes and c, c = 1 reproductions, random floor) prints "STAGE 0 DONE", then 40 states, "STAGE 1 DONE";
# a filed stop logs "STOP: ..." and exits 1. Resumes from results/m67.json.
# Inputs on NFS (gitignored): experiments/{regating_table,keydependent_elicit,keydependent_patch,performed_anchor,open_pair,
# open_pair_instruments,induction_cue,refit_96key}/results/large/, artifacts/vpd/. Writes experiments/size_matched/results/m67.json.
# Expected on an L40: ~15 min (E143 ~56 states in 9.6 min, E172 37 carrying sets in ~5 min); the job stops itself if
# its projection after four states passes 45 min. 40G as refit_96key.sbatch, which loads three of these four readers.
#SBATCH --job-name=commute-size-matched
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/refit_96key/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data compare-lead"; exit 1; }
.venv/bin/python experiments/size_matched/run.py
