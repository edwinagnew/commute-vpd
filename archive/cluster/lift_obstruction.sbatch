#!/bin/bash
# Lift test stage 0 (CPU linear algebra, no forward passes): obstruction fraction, ranks, conditioning.
# Submit from the controller:  cd ~/commute-agents/<role> && sbatch cluster/lift_obstruction.sbatch
#SBATCH --job-name=commute-lift-obstruction
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/lift_test/run.py obstruction
