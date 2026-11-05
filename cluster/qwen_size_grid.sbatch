#!/bin/bash
# E91 (qwen_size_grid) on one cluster L40, debug QoS, $0: experiments/qwen_size_grid/run.py. Rules:
# experiments/qwen_size_grid/results/predictions.json. Submit from the clone root: sbatch cluster/qwen_size_grid.sbatch
# Time: 57 states, about 20 s each on an L40 (job 52219); then matched.py, under 45 min by its docstring. A lost job resumes.
# Exit 0 done; 1 a control missed (results/stopped.json) or reads are missing.
#SBATCH --job-name=commute-qwen_size_grid
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:10:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_size_grid/run.py && .venv/bin/python experiments/qwen_size_grid/matched.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
