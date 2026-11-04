#!/bin/bash
# E81 (qwen_tofu) on one cluster L40, debug QoS, $0: experiments/qwen_tofu/run.py. Rules:
# experiments/qwen_tofu/results/predictions.json. Submit from the clone root:
#   sbatch --time=00:30:00 cluster/qwen_tofu.sbatch --stage 0
#   sbatch cluster/qwen_tofu.sbatch --stage T --rate 1e-5      (and 2e-5)
#   sbatch --time=00:40:00 --dependency=afterok:<T's id> cluster/qwen_tofu.sbatch --stage G --rate 1e-5
# Exit 0 done; 1 a stop (results/stopped.json) or a refusal.
# --mem: E78 peaked at about 14 GB on the host; job 51323 waited 10 min asking for 64G with 4 cards free and 42 GB unallocated.
#SBATCH --job-name=commute-qwen_tofu
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:50:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_tofu/run.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
