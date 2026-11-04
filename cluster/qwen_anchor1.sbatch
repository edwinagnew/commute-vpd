#!/bin/bash
# E78 (qwen_anchor1) on one cluster L40, debug QoS, $0: experiments/qwen_anchor1/run.py. Rules:
# experiments/qwen_anchor1/results/predictions.json. Submit from the clone root, in this order; not submitted by the code writer:
#   sbatch cluster/qwen_anchor1.sbatch T                                  # about 27 min
#   sbatch --dependency=afterok:<T's id> cluster/qwen_anchor1.sbatch G    # about 10 min
# Exit 0 done; 1 a stop (results/stopped.json, every stage refuses until it is removed) or a refusal. A lost T starts
# again: same command. --mem: E61's run of this arm peaked at 11 GB on the host, and δW doubles to 5.64 GB here; 64G as
# its sbatch.
#SBATCH --job-name=commute-qwen_anchor1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_anchor1.sbatch T|G}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_anchor1/run.py --stage "$stage"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
