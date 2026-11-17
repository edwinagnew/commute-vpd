#!/bin/bash
# E128 (qwen_post_cutoff) stage 1 then stage 2 (pools, CPU): novelty sampling on one L40, debug QoS, $0; then results/pools.json.
# Needs results/{predictions.json,stage0.json (passed, with candidates),large/candidates.json} and the results/large link
# (fleet/link-data). Submit from the clone root. Exit 0 done, 2 a filed check missed (results/stage1.json names it),
# 3 paused at a shard boundary: resubmit the same command (fleet/resubmit --code 3 ...). Fable (d): ~128,000 answers of
# ~20 tokens, 15-45 min on an L40. --mem 24G: the bf16 model and the shard files are small (E120 stage 1 and E45's control).
#SBATCH --job-name=commute-e128_stage1
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
.venv/bin/python experiments/qwen_post_cutoff/sample.py "$@"
status=$?
if [ "$status" -eq 0 ]; then
  .venv/bin/python experiments/qwen_post_cutoff/pools.py
  status=$?
fi
echo "=== FINISHED (exit $status) ==="
exit $status
