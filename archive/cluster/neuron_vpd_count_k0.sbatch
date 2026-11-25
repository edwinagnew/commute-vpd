#!/bin/bash
# E117 follow-up (independent check fix 2), description only: on the six E53 step-300 states (lowercase, cipher, broad
# cipher, two seeds) at position 63, the largest window-mean importance change, the counts with mean change > 0.1 / 0.2 /
# 0.3, and the counts with change > 0.3 on >= 10% and >= 50% of windows. W0 control first on each task (all zeros, else
# exit 1). Reuses experiments/neuron_vpd_count/run.py's loaders and sha256 check.
#   cd ~/commute-agents/<role> && sbatch cluster/neuron_vpd_count_k0.sbatch
# Rules: experiments/neuron_vpd_count/results/predictions_k0.json (committed unchanged, else the script refuses).
# Writes experiments/neuron_vpd_count/results/k0_read.json (small) only; commits nothing.
# Expected: a few minutes (two Gamma passes per batch, 16 batches per state, 3 W0 tasks + 6 states).
#SBATCH --job-name=commute-neuron-vpd-count-k0
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/neuron_vpd_count/k0_read.py "$@"
