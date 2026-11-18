#!/bin/bash
# E209 (cell P2.38 run 1): the premise run of the real-text task set on the 67M. Plan:
# reports/plans/2026-11-18_plan_p2-38-run1.md. Rules: experiments/realistic_tasks/results/predictions.json (run.py
# refuses unless it is committed unchanged).
#   cd <clone> && ~/commute-agents/fleet/link-data <role or .> && sbatch --comment=E209 cluster/realistic_tasks_r1.sbatch
# Stage 0 (E53 re-read and zero-update controls, alone), stage 1 (W0 premises, downloads), stage 2 (rate grid, T1 at
# E53's recipe first); each prints "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1.
# Inputs: E53's saved states and E136/E143 records (read only), cached Pile rows (artifacts/sets/{train_0_4096,
# val_0_256}.npy), the model in artifacts/vpd/; downloads inside the job: AG News (19 MB), SST-2 (3 MB), Wikipedia
# fr/sw/tl/yo streamed (first ~1.3M tokens each). Writes only experiments/realistic_tasks/results/stage{0,1,2}.json.
# Expected on an L40: about 20 min (model load 2; stage 0 3: E143 read 56 states in 10 min, plus two E53 share reads;
# stage 1 about 5 incl. downloads; stage 2 about 8: 12 LM runs at ~18 s per 300 steps (E53 log) and 9 label runs);
# stage 2 stops itself if projected past 15 min. Host memory: E53's two gate sets in the share read, ~20 GB (as
# share_untrained.sbatch), so 40G.
#SBATCH --job-name=commute-realistic-r1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/realistic_tasks/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/realistic_tasks/run.py
