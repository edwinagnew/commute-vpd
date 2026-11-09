#!/bin/bash
# E128 (qwen_post_cutoff) novelty check N1: experiments/qwen_post_cutoff/sample.py --think on one L40, debug QoS, $0.
# Needs results/{predictions.json,pools.json} and the results/large link. Submit from the clone root. Exit 0 done, 2 the first
# shard's mean thinking length is over 2,000 tokens (N1 then moves to E120's pod engine, Fable (d); results/n1.json names it),
# 3 paused at a shard boundary: resubmit the same command (fleet/resubmit --code 3 ...). Fable (d): 0.5-1 h on HF if the
# model thinks 1-2k tokens over an unknown fact. --mem: as stage 1.
#SBATCH --job-name=commute-e128_n1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
test -L experiments/qwen_post_cutoff/results/large || { echo "results/large is not linked: run fleet/link-data"; exit 1; }
set +e
.venv/bin/python experiments/qwen_post_cutoff/sample.py --think "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
