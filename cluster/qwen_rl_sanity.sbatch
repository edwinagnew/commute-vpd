#!/bin/bash
# E69 (qwen_rl): the $0 sanity checks on one cluster L40, before the paid run. Submit from the clone root:
#   sbatch cluster/qwen_rl_sanity.sbatch
# Rules in experiments/qwen_rl/results/predictions_sanity.json; results in results/sanity.json and sanity_samples.txt, every
# number also in this job's log. Needs no key: it uploads nothing. Exit 0 every check holds; 1 a check failed or a refusal.
# --mem as cluster/qwen_rl.sbatch: E45's T peaked at 19 GB on the host; δW in fp32 is read back once more here, 5.6 GB.
#SBATCH --job-name=commute-qwen_rl_sanity
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
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_rl/sanity.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
