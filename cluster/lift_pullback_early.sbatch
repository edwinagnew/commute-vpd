#!/bin/bash
# Lift test: faithfulness of the selected pull-back at steps 1, 2, 3, 5, 10 (reported, not scored).
# Submit from the controller:  cd ~/commute-agents/<role> && sbatch cluster/lift_pullback_early.sbatch
#SBATCH --job-name=commute-lift-pullback-early
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

.venv/bin/python experiments/lift_test/run.py pullback_early
