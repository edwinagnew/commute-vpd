#!/bin/bash
# E92 (qwen_comparators) on one cluster L40, debug QoS, $0: experiments/qwen_comparators/run.py. Rules:
# experiments/qwen_comparators/results/predictions.json. Submit from the clone root:
#   sbatch cluster/qwen_comparators.sbatch --states anchor1_s0:100
# Time: 2.6 to 3.1 min per state, 1.8 on anchor 1's 64 rows; the first job's controls about 4 min, anchor 1's step 100 among them. Built
# on experiments/qwen_size_grid/results/run.log: lines 7-8 (16,384 tokens in under 4.8 s with the file already read), lines
# 5-6 (24 s a state with a 5.64 GB file over NFS). A state is about 557,000 tokens (278,000 on anchor 1's rows); the plan has
# the count. A lost job resumes from results/states/. A long list at this limit is cut and resubmitted, losing one state.
# --mem as E91's sbatch, whose log read a peak host RSS of 10.5 GB (cluster/logs/commute-qwen_size_grid-52219.out).
# Exit 0 done; 1 a stop (results/stopped.json) or a refusal.
#SBATCH --job-name=commute-qwen_comparators
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_comparators/run.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
