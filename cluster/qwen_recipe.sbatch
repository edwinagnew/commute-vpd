#!/bin/bash
# E61 (qwen_recipe) stage A on one cluster L40 (44 GiB), debug QoS, $0, about 20 min. Plan:
# reports/reviews/2026-11-02_review_qwen-finetune-recipes.md, section 5; rules in
# experiments/qwen_recipe/results/predictions.json. Submit from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_recipe.sbatch A
# Exit 0 done; 1 an early stop (results/stopped.json); 3 the 25-min budget was reached (resubmit the same command;
# results/A.json skips what is done). --mem: E45's fp32 jobs peak at 11 GB on the host, and the node's RAM is shared
# (cluster/README.md: measured peak plus a margin), so 32G.
#SBATCH --job-name=commute-qwen_recipe
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_recipe.sbatch A}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_recipe/run.py --stage "$stage"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
