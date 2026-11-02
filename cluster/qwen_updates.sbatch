#!/bin/bash
# E45 (qwen_updates) Job 1 on one cluster L40 (46 GB), debug QoS, $0. Plan: reports/plans/2026-10-28_plan_qwen-q2.md;
# code experiments/qwen_updates/run.py; job order and first-minute values in experiments/qwen_updates/RUN_BRIEF.md.
# Submit from the clone root; every argument goes to run.py, e.g.
#   sbatch --mem=32G cluster/qwen_updates.sbatch --stage control
#   sbatch --mem=48G cluster/qwen_updates.sbatch --stage T --arm gold --seed 0
# Exit 0 done; 2 a first-minute check missed (STOP, results/<stage>.json "stopped"); 3 the 25-min budget was reached
# (resubmit the same command: P, S and E resume where they stopped). Large files go to experiments/qwen_updates/results/
# large/, a link into commute-data in the lead's clone (NFS, so later jobs on other nodes read earlier ones' files).
#SBATCH --job-name=commute-qwen_updates
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_updates/run.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
