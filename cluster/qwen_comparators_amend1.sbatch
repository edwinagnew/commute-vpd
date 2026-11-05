#!/bin/bash
# E92, Amendment 1 on one cluster L40, debug QoS, $0: experiments/qwen_comparators/amend1.py. Rules:
# experiments/qwen_comparators/results/predictions_amend1.json. Submit from the clone root, no arguments:
#   sbatch cluster/qwen_comparators_amend1.sbatch
# Time: 9 to 14 min. From cluster/qwen_comparators.sbatch's header: 16,384 tokens in under 4.8 s, 24 s to read a state's file, and
# 1.8 min a state on anchor 1's rows with the layer sweep. A facts site is 24 s + 7 reads of 16,384 tokens, about 1 min, and there are
# four; an anchor 1 control is half that in tokens, about 45 s, and there are two, with the zero state at 30 s; the four earlier
# steps are at most 1.8 min each (results/run.log read 50 s for one). W0's temperature grid is unmeasured: allow 1 min in all.
# A lost job resumes from results/amend1/. Limits as qwen_comparators.sbatch.
# Exit 0 done; 1 a stop (results/stopped_amend1.json) or a refusal.
#SBATCH --job-name=commute-qwen_comparators_amend1
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
.venv/bin/python experiments/qwen_comparators/amend1.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
