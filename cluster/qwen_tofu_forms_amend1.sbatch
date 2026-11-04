#!/bin/bash
# E84 Amendment 1 (qwen_tofu_forms, 10 epochs) on one cluster L40, debug QoS, $0: experiments/qwen_tofu_forms/amend1.py. Rules:
# experiments/qwen_tofu_forms/results/predictions_amend1.json. Submit from the clone root:
#   sbatch cluster/qwen_tofu_forms_amend1.sbatch                 (T then G; a stage already done is skipped)
#   sbatch cluster/qwen_tofu_forms_amend1.sbatch --stage G       (one stage)
# Exit 0 done; 1 a stop (results/stopped_amend1.json) or a refusal. T about 55 min (2500 steps at ~1.3 s), G about 5 min, 14 saves x 20 s.
# --mem: E78 peaked at about 14 GB on the host; job 51323 waited 10 min asking for 64G with 4 cards free and 42 GB unallocated.
#SBATCH --job-name=commute-qwen_tofu_forms_amend1
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
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_tofu_forms/amend1.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
