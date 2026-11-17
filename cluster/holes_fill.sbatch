#!/bin/bash
# E143 (cell P2.10): steering in the fair form and the neuron version on every scorecard pair with saved updates where either
# is missing, beside the share on the same rows. Plan: reports/plans/2026-11-10_plan_E143-holes.md. Rule:
# experiments/holes_fill/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E143 cluster/holes_fill.sbatch
# Stage 0 (controls, alone) then stage 1 (56 saved updates, cross-application, the score); each prints "STAGE n DONE" after
# its results file. A filed stop prints "STOP: ..." and exits 1; stage 1 stops itself if the projected job time passes 25 min.
# Inputs on NFS (gitignored), read only: the updates and windows named in experiments/share_untrained/results/inventory.json,
# E53's cached Pile rows (artifacts/sets/{train_0_4096,val_*}.npy), the model in artifacts/vpd/. Writes only
# experiments/holes_fill/results/stage{0,1}.json; nothing large is saved.
# Expected on an L40: about 15 min (model load 2; stage 0 2; stage 1 about 10: 50 steered updates on 75 to 500 read rows and
# up to 512 selection rows, E140's 12 took 5 min on 9,600 fitting windows). Host memory: one update (113 MB) at a time and
# E53's windows; E136's 40 GB was for the decomposition's gates on E53's rows, which this job reads only on two kv states.
#SBATCH --job-name=commute-holes-fill
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
test -f experiments/holes_fill/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/holes_fill/run.py
