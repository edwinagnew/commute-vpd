#!/bin/bash
# E158 (cell P2.21): E98's neuron reading (r_n, r_half) on the case-label and repeat-label real/twin pairs, in the label
# contrast, to fill the method table's four empty neuron cells. Rules: experiments/neuron_twins/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E158 cluster/neuron_twins.sbatch
# Stage 0 (zero update, last-block down_proj control) then stage 1 (12 states); each prints "STAGE n DONE" after
# results/cells.json; a filed stop prints "STOP: ..." and exits 1.
# Inputs on NFS (gitignored): experiments/{case_label,fresh_reuse_pair}/results/large/{task.pt,dW_{real,twin}_s{0,1}.pt},
# experiments/real_vs_twin/results/large/dW_{case,repeat}_{real,twin}_s2.pt, artifacts/vpd/. Writes only
# experiments/neuron_twins/results/cells.json (small).
# Expected on an L40: about 4 min. Model load ~1-2 min (E140, cluster/logs/commute-real-vs-twin-57752.out); 18 reads of
# 256 rows, 7 forwards per batch of 64: seconds each. E98 read 67 states, its controls' fine-tunes and a top-k grid in
# 4 min (cluster/logs/commute-neuron-r-52786.out, peak GPU 12.1 GB, host RSS 6.1 GB).
#SBATCH --job-name=commute-neuron-twins
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --comment=E158
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/neuron_twins/results/predictions.json || { echo "file the predictions first"; exit 1; }
test -L experiments/real_vs_twin/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data compare-lead"; exit 1; }
.venv/bin/python experiments/neuron_twins/run.py
