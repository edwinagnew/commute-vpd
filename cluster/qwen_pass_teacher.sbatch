#!/bin/bash
# E82: E76's clean-loss state read through E68's reading of u1. About 5 min to load and 3 min to read.
#   cd ~/commute-agents/<role> && sbatch cluster/qwen_pass_teacher.sbatch
#SBATCH --job-name=commute-e82
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_pass_test/teacher_read.py --decomp "$COMMUTE_QWEN_DIR"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
