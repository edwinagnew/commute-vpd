#!/bin/bash
# E64 capped, exploratory and reported only: the saved capped readout through the possession rule per block on W0 and the
# scored run's windows, with no fine-tuned state; filing experiments/readout64/results/predictions_capped.json, which the
# script refuses to run without; writes results/capped.json and commits nothing; expected about 1 min, $0.
#   cd ~/commute-agents/<role> && sbatch cluster/readout64_capped.sbatch
#SBATCH --job-name=commute-readout64-capped
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/readout64/capped.py
