#!/bin/bash
# E128 (qwen_post_cutoff) stage 0, the control: experiments/qwen_post_cutoff/sample.py --control on one L40, debug QoS, $0.
# Needs results/predictions.json; results/large/candidates.json for parts (b) and the yield read (a missing file skips them, a test
# of the file before the pools exist; stage 1 refuses without a passed stage 0 with candidates). Submit from the clone root.
# Exit 0 passed, 2 a band missed (results/stage0.json names it, nothing else runs). --mem: bf16 copy 3.4 GB + the fp32 read
# of GSM8K (E45's control peaked at 11 GB on the host), so 24G.
#SBATCH --job-name=commute-e128_stage0
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_post_cutoff/sample.py --control "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
