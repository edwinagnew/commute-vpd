#!/bin/bash
# kv_task smoke: all stages with --smoke.
# Submit from the controller:  cd ~/commute-agents/<role> && sbatch cluster/kv_smoke.sbatch
#SBATCH --job-name=commute-kv-smoke
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/kv_task/results/predictions.json
echo "$(date +%T) start"
.venv/bin/python experiments/kv_task/run.py all --smoke
echo "$(date +%T) done"
