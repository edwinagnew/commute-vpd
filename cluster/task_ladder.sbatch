#!/bin/bash
# Task ladder (round 3). Plan: reports/2026-10-15_plan_task-ladder.md. One stage per job, chained, so that an overrun of
# one stage never loses another (the three together sit near the 2 h debug limit):
#   cd ~/commute-agents/<role>
#   J1=$(sbatch --parsable cluster/task_ladder.sbatch noise)
#   J2=$(sbatch --parsable --dependency=afterok:$J1 cluster/task_ladder.sbatch mine)
#   J3=$(sbatch --parsable --dependency=afterok:$J2 cluster/task_ladder.sbatch ladder)
#   echo "$J1 $J2 $J3"   # record all three in the analysis report
# Before that: `cluster/sync.sh up`, predictions filed (results/predictions.json), and a smoke run on a GPU node:
#   srun --partition=compute --account=c11 --qos=debug --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:20:00 \
#     bash -c 'source cluster/env.sh && cd $COMMUTE_REPO && .venv/bin/python experiments/task_ladder/run.py all --smoke'
# Expected: noise ~30 min (3 Pile seeds + 4 short pull-backs), mine ~15 min, ladder ~60 min (7 tasks + 2 Pile checks).
#SBATCH --job-name=commute-task-ladder
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/task_ladder/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
STAGE="${1:?stage required: noise | mine | ladder}"
.venv/bin/python experiments/task_ladder/run.py "$STAGE"
