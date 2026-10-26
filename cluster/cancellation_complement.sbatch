#!/bin/bash
# Complement revert (E20) on E11's saved states. Plan: reports/2026-10-21_plan_cancellation-complement.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/cancellation_complement.sbatch all    # ~12 min: W0 (gate first) and eight states, 128 rows
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Inputs on NFS:
# experiments/reading_a_coverage/results/large/<task>/{pullback_s0/state_step10.pt, dW_ft_step10_s0.pt},
# experiments/transport_test/results/large/{alive.pt, windows.pt}; experiments/cancellation/results/cancellation.json.
#SBATCH --job-name=commute-complement
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/cancellation_complement/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/cancellation_complement/run.py "$@"
