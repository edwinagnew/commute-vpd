#!/bin/bash
# Lift test, amendment 1: prediction 2' (lift_faith) then prediction 3' (pullback2).
# Submit from the controller:  cd /mnt/nw/home/e.agnew/commute && sbatch cluster/lift_amend.sbatch
#SBATCH --job-name=commute-lift-amend
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/lift_test/run.py lift_faith
.venv/bin/python experiments/lift_test/run.py pullback2
