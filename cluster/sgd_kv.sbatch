#!/bin/bash
# sgd_kv: SGD and low-rate Adam component training on E10's key-to-value tasks, one task per shard
# (reports/plans/2026-10-21_plan_sgd-kv.md). About 22 min per shard on an L40.
#   cd ~/commute-agents/<role>
#   sbatch cluster/sgd_kv.sbatch --shard 0                              # rand32 first and alone (its control must read 'fail')
#   for n in 1 2; do sbatch cluster/sgd_kv.sbatch --shard $n; done      # once shard 0 is past its control
# Shards: 0 rand32, 1 ident32, 2 rand1.
# Needs on NFS: experiments/kv_task/results/kv_task.json, experiments/kv_task/results/large/windows_<task>.pt,
# experiments/kv_task/results/large/<task>/pullback/state_step<n>.pt (E10's Adam 1e-5 states; regenerated if absent),
# experiments/transport_test/results/large/{alive,windows}.pt, ../windows.json.
# Before that: `cluster/sync.sh up` and predictions filed (results/predictions.json).
#SBATCH --job-name=commute-sgd-kv
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
test -f experiments/sgd_kv/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/sgd_kv/run.py "$@"
