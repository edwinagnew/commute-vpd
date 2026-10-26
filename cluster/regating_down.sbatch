#!/bin/bash
# What the turned-down half of re-gating's fixed change does, read at W0 (E29). Plan: reports/plans/2026-10-22_plan_regating-down.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_down.sbatch      # ~3-5 min on an L40: the J_nat control first, stops the job on a fail
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Inputs on NFS (gitignored):
# experiments/regating_leak/results/large/{emoticon,r1_t534_v369,r2_t359_v816}_s{0,1}_shift.pt,
# experiments/regating_tasks/results/large/<task>_shift.pt (E28; optional, missing ones are skipped),
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt},
# experiments/task_ladder/results/large/windows_<task>.pt; Pile val rows 0..600 (data cache).
#SBATCH --job-name=commute-regating-down
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/regating_down/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/regating_down/run.py "$@"
