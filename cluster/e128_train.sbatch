#!/bin/bash
# E128 (qwen_post_cutoff) stage 3 on one L40, $0: experiments/qwen_post_cutoff/train.py, one arm per job; arms may run at the same
# time (each has its own files). Needs results/{predictions.json,pools.json} and the results/large link (fleet/link-data).
# Submit from the clone root, one per arm:
#   sbatch cluster/e128_train.sbatch known 0      new 0      known 1      new 1
# Debug QoS is the default (2 h a job); the budget leaves 10 min for the last unit and the exit. Exit 0 done; 1 a stop
# (results/stopped_<arm>.json) or a refusal; 3 paused, the work so far saved: resubmit the same command (fleet/resubmit --code 3).
# --mem: E120's arms' peak (24G), one more fp32 dW than E61's 11 GB. Extra arguments go to train.py (--t-chunk N).
#SBATCH --job-name=commute-e128_train
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
pool=${1:?usage: sbatch cluster/e128_train.sbatch known|new 0|1 [train.py arguments]}
seed=${2:?usage: sbatch cluster/e128_train.sbatch known|new 0|1 [train.py arguments]}
shift 2
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$pool $seed $*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
test -L experiments/qwen_post_cutoff/results/large || { echo "results/large is not linked: run fleet/link-data"; exit 1; }
set +e
.venv/bin/python experiments/qwen_post_cutoff/train.py --pool "$pool" --seed "$seed" --budget-min 110 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
