#!/bin/bash
# kv_task scored run: all stages.
# Submit from the controller:  cd /mnt/nw/home/e.agnew/commute && sbatch cluster/kv_run.sbatch
#SBATCH --job-name=commute-kv-run
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/kv_task/results/predictions.json
echo "$(date +%T) start"
.venv/bin/python experiments/kv_task/run.py all
echo "$(date +%T) done"
