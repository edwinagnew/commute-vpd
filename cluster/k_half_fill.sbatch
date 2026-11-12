#!/bin/bash
# E172 (cell P2.24): E98's k_half on the built-pair states it never read (E54 and C3 matched ciphers, E147's 24 refit
# 96-key updates). Rules: experiments/k_half_fill/results/predictions.json (run.py refuses unless committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E172 cluster/k_half_fill.sbatch
# Stage 0 (c3_rank1_s0, lowercase_s0 against E98's saved k_half) then stage 1 (37 states), then the cells; each prints
# "STAGE n DONE" / "CELLS DONE" after results/cells.json; a filed stop logs "FAIL: ..." and exits 1.
# Inputs on NFS (gitignored): E98's (experiments/{regating_table,keydependent_elicit,keydependent_patch,performed_anchor,
# open_pair,open_pair_instruments,induction_cue}/results/large/), experiments/refit_96key/results/large/dW_*.pt,
# artifacts/vpd/. Writes only experiments/k_half_fill/results/cells.json (small).
# Expected on an L40: about 5 min. E98's whole job (67 states' cells, 43 carrying sets, two fine-tunes) took 4 min
# (cluster/logs/commute-neuron-r-52786.out): load ~1 min, KV carrying sets ~1 s each, lowercase ~14 s; peak GPU 12.1 GB,
# host RSS 6.1 GB.
#SBATCH --job-name=commute-k-half-fill
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --comment=E172
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/k_half_fill/results/predictions.json || { echo "file the predictions first"; exit 1; }
test -L experiments/refit_96key/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data compare-lead"; exit 1; }
.venv/bin/python experiments/k_half_fill/run.py
