#!/bin/bash
# sgd_ladder: SGD and low-rate Adam component training on E11's ladder tasks at step 10, one task per shard
# (reports/plans/2026-10-21_plan_sgd-ladder.md). Runs only after sgd_faith's control has passed.
#   cd ~/commute-agents/<role>
#   sbatch cluster/sgd_ladder.sbatch --shard 0                                # emoticon first and alone, ~22 min on an L40
#   for n in 1 2 3 4 5; do sbatch cluster/sgd_ladder.sbatch --shard $n; done  # once shard 0 has finished clean
# Shards: 0 emoticon, 1 r0_t352_v310, 2 r1_t534_v369, 3 r1_t588_v452, 4 r2_t275_v2667, 5 r2_t359_v816.
# Needs on NFS: experiments/reading_a_coverage/results/large/<task>/pullback_s0/state_step10.pt (E11's Adam 1e-5 states),
# experiments/task_ladder/results/large/windows_<task>.pt, experiments/transport_test/results/large/{alive,windows}.pt.
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json).
#SBATCH --job-name=commute-sgd-ladder
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/sgd_ladder/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/sgd_ladder/run.py "$@"
