#!/bin/bash
# Re-gating patterns on small updates (E24). Plan: reports/plans/2026-10-22_plan_small-update-reruns.md -> E24 (Agent notes).
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_small.sbatch     # ~5-8 min on an L40: G1 (large state vs E18) first, stops the job on a fail
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Inputs on NFS:
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
# experiments/task_ladder/results/large/windows_r1_t534_v369.pt.
#SBATCH --job-name=commute-regating-small
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
test -f experiments/regating_small/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/regating_small/run.py "$@"
