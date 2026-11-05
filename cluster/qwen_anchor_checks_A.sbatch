#!/bin/bash
# E88 (qwen_anchor_checks) stage A on one cluster L40, debug QoS, $0: experiments/qwen_anchor_checks/run.py. Rules:
# experiments/qwen_anchor_checks/results/predictions.json. Submit from the clone root:
#   sbatch cluster/qwen_anchor_checks_A.sbatch
# Time: training at most 1629 s on an L40 (E78's train.json, with 8 saves and more reads), reads 229 s (E78's G). A lost job resumes at the reads.
# Exit 0 done or already done; 1 a stop (results/stopped_A.json) or a refusal.
# --mem as E84's sbatch: E78 peaked at about 14 GB on the host; job 51323 waited 10 min asking for 64G with 4 cards free.
#SBATCH --job-name=commute-qwen_anchor_checks_A
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:10:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_anchor_checks/run.py --stage A
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
