#!/bin/bash
# What the fixed change is made of, on E28's eight states (E32). Plan: reports/2026-10-22_plan_shift-anatomy.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/shift_anatomy.sbatch     # ~4-8 min on an L40: G1 (emoticon vs E28) first, stops the job on a fail
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Gitignored inputs on NFS:
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt};
# experiments/task_ladder/results/large/windows_{r0_t352_v310,r0_t588_v320,r1_t588_v452,r2_t275_v2667}.pt;
# experiments/kv_task/results/large/windows_{ident32,rand32,rand1}.pt; experiments/regating_tasks/results/regating_tasks.json
# (in git). The Pile stage reads val rows 0..600 (data.pile_rows; cached under $COMMUTE_CACHE_DIR/sets after E24/E29).
#SBATCH --job-name=commute-shift-anatomy
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
trap 'echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/shift_anatomy/results/shift_anatomy.json, $COMMUTE_REPO/experiments/shift_anatomy/results/large/full.log ==="' EXIT
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/shift_anatomy/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/shift_anatomy/run.py "$@"
