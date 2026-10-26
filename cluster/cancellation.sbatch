#!/bin/bash
# Cancellation tests (E17) on E11's saved states. Plan: reports/2026-10-21_plan_cancellation.md.
#   cd ~/commute-agents/<role>
#   sbatch --gres=none --cpus-per-task=16 cluster/cancellation.sbatch all --smoke --device cpu   # CPU smoke, no GPU queue
#   sbatch cluster/cancellation.sbatch all --smoke    # GPU smoke, ~5 min: two tasks, 8 rows, 2 adversary steps
#   sbatch cluster/cancellation.sbatch all            # ~40 min: nine states, 128 rows, 20-step adversary per state
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Inputs on NFS:
# experiments/reading_a_coverage/results/large/<task>/{pullback_s0/state_step10.pt, dW_ft_step10_s0.pt},
# experiments/transport_test/results/large/{alive.pt, windows.pt}.
#SBATCH --job-name=commute-cancellation
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/cancellation/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/cancellation/run.py "$@"
