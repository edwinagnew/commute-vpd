#!/bin/bash
# Mechanism or leak (E18). Plan: reports/2026-10-21_plan_regating-leak.md (build notes there).
#   cd ~/commute-agents/<role>
#   sbatch cluster/regating_leak.sbatch     # ~12 min on an L40: G1 first (stops the job on a fail), three tasks, Pile stages
# No separate smoke: G1 (reproduce r, W0's gates recover 0) runs first and alone and stops the job in its first minutes.
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json). Inputs on NFS:
# experiments/reading_a_coverage/results/large/{emoticon,r1_t534_v369,r2_t359_v816}/dW_ft_step10_s0.pt,
# experiments/task_ladder/results/large/windows_{r1_t534_v369,r2_t359_v816}.pt,
# experiments/transport_test/results/large/{alive.pt, windows.pt}, experiments/transport_test/results/windows.json.
#SBATCH --job-name=commute-regating-leak
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
test -f experiments/regating_leak/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/regating_leak/run.py "$@"
