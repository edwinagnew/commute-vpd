#!/bin/bash
# E199 (cell P2.32): a built pair where the prior and strength-1 steering disagree, by a construction other than E151's
# common-word cipher; the share read on it. Plan: reports/plans/2026-11-17_plan_p2-32-disagreeing-pairs.md. Rule:
# experiments/disagreeing_pairs2/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E199 cluster/disagreeing_pairs2.sbatch
# Stage 0 (E151's controls, then four tables and their priors; no training), stage 1 (in order, 3 fine-tunes and strength-1
# steering per construction until the first reads 'opposite'), stage 2 (the share on the pair's 3 new states, the reading);
# each prints "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1 (stage 0 or 1: the pair
# could not be built, which closes the cell); stage 1 stops itself if the projected job time passes 25 min after four states.
# Inputs on NFS, read only: E147's inventory, stage0.json and 15 refits (experiments/refit_96key/results/large/dW_{copy,elicit,
# rank1,class_shuffled,prior_cipher}_s{0,1,2}.pt, sha256 checked), E149's stage1.json, prior_vs_share's stage2.json, E143's
# predictions.json, open_pair's amend3.json and windows, E151's predictions_amend1.json (read when amend1.py loads), the
# model in artifacts/vpd/, the Pile rows keydep.vocab reads.
# Writes experiments/disagreeing_pairs2/results/stage{0,1,2}.json and, in results/large/ (a symlink to commute-data), four
# tables' windows (about 3 MB each) and at most 12 bf16 updates dW_<construction>_s{0,1,2}.pt (about 57 MB each).
# Expected on an L40: about 8 min (load 2, controls 2, tables 1, stage 1 at most 2, stage 2 0.5). Memory: E147, E149 and
# E151, the same readers on the same maps, peaked at 10.6 GB GPU and 6.1 GB host with 24G asked.
#SBATCH --job-name=commute-disagreeing-pairs2
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
test -f experiments/disagreeing_pairs2/results/predictions.json || { echo "file the predictions first"; exit 1; }
test -d experiments/disagreeing_pairs2/results/large/ || { echo "results/large is missing: run fleet/link-data"; exit 1; }
.venv/bin/python experiments/disagreeing_pairs2/run.py
