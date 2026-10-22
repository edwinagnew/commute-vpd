#!/bin/bash
# sgd_ladder_long: SGD (3a's rate), Adam 1e-5 and Adam 3e-6 component training to step 1000 on emoticon and three ladder
# tasks, Pile harness at steps 100 and 1000, one task per job (reports/2026-10-21_plan_sgd-ladder-long.md).
#   cd /mnt/nw/home/e.agnew/commute
#   sbatch cluster/sgd_ladder_long.sbatch --task emoticon          # the control, first and alone; ~41 min on an L40
#   for t in r0_t352_v310 r1_t534_v369 r2_t275_v2667; do sbatch cluster/sgd_ladder_long.sbatch --task $t; done
#                                                                 # once emoticon's G1, C1 and C2 have passed
# Needs on NFS: experiments/sgd_ladder/results/sgd_ladder_shard*.json (3a), experiments/split_comparison/results/
# split_comparison.json (E16), experiments/task_ladder/results/large/windows_<task>.pt,
# experiments/transport_test/results/large/{alive,windows}.pt and results/windows.json.
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json).
#SBATCH --job-name=commute-sgd-ladder-long
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:05:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/sgd_ladder_long/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/sgd_ladder_long/run.py "$@"
