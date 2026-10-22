#!/bin/bash
# H' kill tests on seven new tasks (E28). Plan: reports/2026-10-22_plan_regating-tasks.md.
#   cd /mnt/nw/home/e.agnew/commute
#   sbatch cluster/regating_tasks.sbatch     # ~4-6 min on an L40: G1 (emoticon vs E18, the moved arms too) first, stops the job on a fail
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Gitignored inputs on NFS:
# experiments/transport_test/results/{windows.json, large/alive.pt, large/windows.pt};
# experiments/task_ladder/results/large/windows_{r0_t352_v310,r0_t588_v320,r1_t588_v452,r2_t275_v2667}.pt;
# experiments/kv_task/results/large/windows_{ident32,rand32,rand1}.pt (NFS only, not on the laptop).
#SBATCH --job-name=commute-regating-tasks
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/regating_tasks/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/regating_tasks/run.py "$@"
