#!/bin/bash
# E147 (cell P2.12): E58's four 96-key pairs fine-tuned again (keydep.finetune, seeds 0-2, read at step 30, kept in bf16), the share
# checked against E58's saved reads, then the untrained decomposition, the neuron forms and fair steering read on them.
# Plan: reports/plans/2026-11-10_plan_e147-refit-96key.md. Rule and tolerances: experiments/refit_96key/results/predictions.json;
# maps: results/inventory.json (run.py refuses unless both are committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E147 cluster/refit_96key.sbatch
# Stage 0 (controls, alone): the readers on states they read before, then the 24 refits, each stopping the job if its share is off
# E58's by more than the map's seed range; stage 1: every reading of the 24 states and the score. Each prints "STAGE n DONE"
# after its file; a stop prints "STOP: ..." and exits 1. A rerun resumes from results/stage{0,1}.json and the saved updates.
# Inputs (NFS, read only): E58's windows in experiments/open_pair/results/large/amend{3,6,7}/, the kept updates named in
# inventory.json, E136's, E98's and E143's results JSONs and the states holes_fill's stage 0 reads, the model in artifacts/vpd/.
# Writes experiments/refit_96key/results/stage{0,1}.json and results/large/dW_<map>_s<seed>.pt (24 x 57 MB, bf16).
# Expected on a debug L40, from measured runs: model load and SVDs about 2 min; holes_fill's stage 0 about 2 min (its plan's
# estimate; its whole job 57802 took about 10 min); 24 fine-tunes of 300 steps at 7.7-8.8 s each (cluster/logs/commute-neuron-r-52786.out, "checkpoint 300" lines) plus a
# save and a share read, about 8 min; stage 1 about 8 min (E143 read 56 states in about 8 min, E98 67 in under 4, E136 projected
# 2.4 min for its whole stage 1). Total about 20 min; the job stops itself past 50 min projected; limit 45 min.
# Memory: E143 and E98 peaked at 6.1 GB host and 12.1 GB GPU (their logs' last lines); E136's 20.4 GB came from E53's gates,
# which this job never reads. 24G as E143.
#SBATCH --job-name=commute-refit-96key
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/refit_96key/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/refit_96key/run.py
