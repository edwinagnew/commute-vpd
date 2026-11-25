#!/bin/bash
# E117, the switched-on set and re-switching on MLP neurons at VPD's count (67M), on E12/E46's copy and random maps and
# E53's lowercase and two ciphers. Plan: reports/plans/2026-11-06_plan_e117-neurons-at-vpd-count.md. Rules:
# experiments/neuron_vpd_count/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/neuron_vpd_count.sbatch
# sha256 of every state and windows file before the model loads. W0 control on all seven tasks first (exact zeros,
# log-odds differences <= 1e-5), then 14 states, random maps first, each checked against E12's S_on count, E46/E53's r
# and E98's r_n before the next (a miss exits 1), then the norm-matched random update; then score.py.
# Inputs on NFS (gitignored): experiments/{regating_table,performed_anchor,keydependent_elicit,transport_test}/results/large/,
# the decomposition in artifacts/vpd/, Pile rows in artifacts/sets/; committed JSON records of E12, E46, E53, E98.
# Writes experiments/neuron_vpd_count/results/{cells,score}.json (small) only; commits nothing.
# Expected: under 20 minutes (plan, "Cost"): per batch 4 Gamma passes, ~23 masked forwards, ~23 restored forwards and
# ~50 fp32 neuron forwards; 2 batches per KV state, 16 per E53 state.
#SBATCH --job-name=commute-neuron-vpd-count
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
test -f experiments/neuron_vpd_count/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/neuron_vpd_count/run.py "$@"
.venv/bin/python experiments/neuron_vpd_count/score.py
