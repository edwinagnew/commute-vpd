#!/bin/bash
# E45 (qwen_updates) Amendment 2 on one cluster L40 (46 GB), debug QoS, $0. Plan:
# reports/plans/2026-10-29_plan_e45-amendment2.md; predictions experiments/qwen_updates/results/predictions_amend2.json;
# job order, first-minute values and stops in experiments/qwen_updates/RUN_BRIEF_amend2.md. Submit from the clone root;
# every argument goes to run.py, e.g.
#   sbatch cluster/qwen_updates_amend2.sbatch --stage P
#   sbatch cluster/qwen_updates_amend2.sbatch --stage T --arm own --seed 0
# Exit 0 done; 2 a first-minute check missed (STOP, results/<stage>.json "stopped"); 3 the 25-min budget was reached
# (resubmit the same command: P, S and E resume where they stopped). Large files go to experiments/qwen_updates/results/
# large/, a link into commute-data. Memory: host peaks measured 11-19 GB (results/run.log), so the default 48G holds.
#SBATCH --job-name=commute-qwen_updates_amend2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
.venv/bin/python experiments/qwen_updates/run.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
