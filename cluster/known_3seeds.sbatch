#!/bin/bash
# E299 (cell P2.77): Fig 4 and App. I on 3 seeds a side. Five seed-2 fine-tunes (e46 ident32, rand32; e53 lowercase,
# cipher; e54 cipher at its own step), each by its siblings' recipe, then every reading the figure shows by the code that
# read the siblings. Card ~/commute-data/cards/E299.md; rule experiments/known_3seeds/results/predictions.json (run.py
# refuses unless it is committed unchanged).
#   cd ~/commute-agents/baseline-lead && ~/commute-agents/fleet/link-data baseline-lead && sbatch --comment=E299 cluster/known_3seeds.sbatch
# Stage 0 (zero update on every reader, two old states read again), stage 1 (recipe controls, then the five fine-tunes,
# saved), stage 2 (readings); each prints "STAGE n DONE" after its file; a stop prints "STOP: ..." and exits 1. A rerun
# resumes stages 1 and 2 from results/stage{1,2}.json and the saved updates.
# Inputs (NFS, read only): the updates and windows named in share_untrained/results/inventory.json (E46, E53, E54, E56,
# E62, case_label), E53's cached Pile rows and a0.json, the saved readings' JSONs, the model in artifacts/vpd/.
# Writes experiments/known_3seeds/results/stage{0,1,2}.json and results/large/ (2 x 113 MB fp32 e46, 2 x 57 MB bf16 e53,
# 1 x 113 MB fp32 e54).
# Expected on a debug L40, from measured runs: model load, SVDs and the five tasks' windows about 3 min (E136 56944,
# E147 57961); holes_fill's stage 0 about 2 min (E147); two old states read by every reader about 1 min (E147: 17 s a kv
# state for E136 + E98 + E143; E143 57802: about 25 s an e53 state's steering); stage 1: 4 kv fine-tunes at about 8 s
# (E147 "checkpoint 300"), 4 e53 fine-tunes at about 18 s (performed_anchor finetune.log elapsed_s 17.5-18.4), 5 control
# share reads at under 10 s, about 3 min; stage 2: 5 states at about 30 s, 3 min. Total about 12-15 min; run.py stops
# itself past 40 min projected; limit 60 min.
# Memory: E136's 20.4 GB host peak came from E53's gates (job 56944, three e53 tasks; this job reads two, one at a time);
# E147/E143 peaked at 6.1 GB host, 10.6-12.1 GB GPU. 40G as share_untrained.sbatch.
#SBATCH --job-name=commute-known-3seeds
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --comment=E299
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/known_3seeds/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/known_3seeds/run.py
