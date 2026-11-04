#!/bin/bash
# E76 Amendment 2 ($0, ~2 min): each arm's saved change on blocks 0-14 and 15-20, on the node that holds E76's g
# (/ephemeral/$USER/e76). Submit from the clone root once L_offsets exists: sbatch cluster/qwen_masked_teacher_amend2.sbatch
#SBATCH --job-name=commute-qwen_masked_amend2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
set +e
.venv/bin/python experiments/qwen_masked_teacher/amend2.py --decomp "$COMMUTE_QWEN_DIR" --out "/ephemeral/$USER/e76" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
