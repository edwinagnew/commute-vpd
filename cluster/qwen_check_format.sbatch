#!/bin/bash
# E45 (qwen_updates) check_format on one cluster L40 (46 GB), debug QoS, $0: is anchor 2's GSM8K collapse lost maths or
# a carried-over answer format? Code and expected values: experiments/qwen_updates/check_format.py. Submit from the
# clone root, after the E45 jobs writing results/ have finished:
#   sbatch cluster/qwen_check_format.sbatch                    # stage P's chosen prompt (results/P.json)
#   sbatch cluster/qwen_check_format.sbatch --prompt recheck   # if P has not chosen
# Exit 0 done; 3 the 25-min budget was reached (resubmit the same command: it resumes at the next state).
#SBATCH --job-name=commute-qwen_check_format
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # an earlier P job OOMed from fragmentation
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_updates/check_format.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
