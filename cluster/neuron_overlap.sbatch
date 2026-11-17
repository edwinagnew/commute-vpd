#!/bin/bash
# E155 (cell P2.18): do the same top-firing neurons stay on after a fine-tune? Top-240 overlap and the share of the firing change
# on neurons in both sets, per saved state of the seven built pairs. Plan: reports/plans/2026-11-11_plan_p2-18-neuron-overlap.md.
# Rule: experiments/neuron_overlap/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E155 cluster/neuron_overlap.sbatch
# Stage 0 (W0 against W0 exactly 1.0, alone, then the positive control), stage 1 (56 saved updates), stage 2 (12 random updates,
# the calls); each prints "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1; stage 1 stops itself
# if the projected job time passes 25 min.
# Inputs on NFS (gitignored), read only: the updates and windows named in experiments/share_untrained/results/inventory.json,
# E53's cached Pile rows, the model in artifacts/vpd/. Writes only experiments/neuron_overlap/results/stage{0,1,2}.json;
# nothing large is saved.
# Expected on an L40: about 8 min (E143, job 57802, cluster/logs/commute-holes-fill-57802.out: the same 56 updates on the same
# rows with steering and four TopK forwards each took 12 min in all, peak host RSS 6.1 GB; this job runs two forwards per
# update, 71 updates in all). Host memory: one update (113 MB) at a time and E53's windows.
#SBATCH --job-name=commute-neuron-overlap
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/neuron_overlap/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/neuron_overlap/run.py
