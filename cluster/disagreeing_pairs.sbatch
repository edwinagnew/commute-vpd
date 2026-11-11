#!/bin/bash
# E151 (cell P2.17): new built pairs where the prior and strength-1 steering disagree; which one the share's direction follows.
# Plan: reports/plans/2026-11-11_plan_p2-17-disagreeing-pairs.md. Rule: experiments/disagreeing_pairs/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E151 cluster/disagreeing_pairs.sbatch
# Stage 0 (controls and the two new tables, alone), stage 1 (6 fine-tunes, strength-1 steering on 18 states), stage 2 (share
# on 18 states, the score); each prints "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1;
# stages 1 and 2 stop themselves if the projected job time passes 25 min after four states.
# Inputs on NFS, read only: E147's inventory, stage0.json and 15 refits (experiments/refit_96key/results/large/dW_{copy,elicit,
# rank1,class_shuffled,prior_cipher}_s{0,1,2}.pt, sha256 checked), E149's stage1.json, prior_vs_share's stage2.json, E143's
# predictions.json, open_pair's amend3.json and windows (amend3/, amend6/, amend7/), the model in artifacts/vpd/.
# Writes experiments/disagreeing_pairs/results/stage{0,1,2}.json and, in results/large/ (a symlink to commute-data), the two
# new maps' windows (about 3 MB each) and six bf16 updates dW_{common_cipher,one_word}_s{0,1,2}.pt (about 57 MB each).
# Expected on an L40: about 10-12 min (appendix of the plan: load 2, stage 0 2, stage 1 2, stage 2 3). Memory: E147 and E149,
# the same readers on the same maps, peaked at 10.6 GB GPU and 6.1 GB host with 24G asked (jobs 57961, 58001).
#SBATCH --job-name=commute-disagreeing-pairs
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
test -f experiments/disagreeing_pairs/results/predictions.json || { echo "file the predictions first"; exit 1; }
test -d experiments/disagreeing_pairs/results/large/ || { echo "results/large is missing: run fleet/link-data"; exit 1; }
.venv/bin/python experiments/disagreeing_pairs/run.py
