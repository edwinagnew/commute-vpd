#!/bin/bash
# E91 Amendment 1 on one cluster L40, debug QoS, $0: experiments/qwen_size_grid/past_line.py. Rules:
# experiments/qwen_size_grid/results/predictions_past_line.json. Submit from the clone root.
# Time: 1.05 s per step (S_train.json, L40) to about step 600, about 2 min per answer set (E88 S), for each seed: about 50 min. A lost job redoes an unfinished seed.
#SBATCH --job-name=commute-qwen_size_grid_past_line
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_size_grid/past_line.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
