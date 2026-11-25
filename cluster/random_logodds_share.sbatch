#!/bin/bash
# EXXX (cell P2.82): a random baseline for the log-odds re-gating share of Fig `known`. Random changes (E228's directions)
# scaled until the trained answer gains one log-odds, the share read on them beside each fine-tune's own. Card
# experiments/random_logodds_share/card.md; settings and expected values experiments/random_logodds_share/results/predictions.json
# (run.py refuses unless it is committed unchanged, and while its "experiment" is still EXXX).
# Stage 0 first and alone; the six shards only after it prints "STAGE 0 DONE" (run.py refuses them before):
#   sbatch --comment=EXXX cluster/random_logodds_share.sbatch control
#   sbatch --comment=EXXX cluster/random_logodds_share.sbatch states 0 6      # ... through states 5 6, in parallel
# A stop prints "STOP: ..." and exits 1. A shard rerun skips finished draws and states (results/states/<state>.json).
# Inputs (NFS, read only): the 48 saved updates named in known_3seeds/results/known3.json (via share_untrained,
# known_3seeds and refit_96key inventories), E136's read windows and E53's cached Pile rows, the model in artifacts/vpd/.
# Writes experiments/random_logodds_share/results/control.json and results/states/*.json (a few KB each).
# Expected on a debug L40, from measured runs: model load, SVDs and windows about 3 min (E299, E136 56944); control 16
# zero reads + 48 fine-tune reads + 64 KL reads at about 2 s (E228 read_seconds 1.6-2.2 s, e53 3.7 s), about 4 min, so
# about 7 min; each shard 8 states x 3 draws x about 16 reads (2 sign probes + E228's 14 evals a match) x 1.8 s, about
# 14 min with the e53 states and KL reads, up to about 24 min when every draw tries both signs, plus 3 min load (plan's
# Cost and time). Limit 30 min; a cut shard resumes.
# Memory: E136's 20.4 GB host peak came from E53's gates (job 56944); 40G as share_untrained.sbatch and known_3seeds.sbatch.
#SBATCH --job-name=commute-random-logodds
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --comment=EXXX
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/random_logodds_share/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/random_logodds_share/run.py "$@"
