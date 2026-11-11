#!/bin/bash
# E163 stage kl-check (P3.4): E163's output-change read on E69's saved R updates (steps 10, 50, 100). Submit from the clone root:
#   sbatch --comment='E163 stage: kl-check' cluster/qwen_rl_kl_check.sbatch
# Results in experiments/qwen_rl_readable/results/kl_check.json, every number also in this job's log.
#SBATCH --job-name=commute-qwen_rl_kl_check
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_rl_readable/kl_check.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
