#!/bin/bash
# cancellation_sgd (E23): the adversary's excess in parts, and the budget sweep, on E21's saved SGD/Adam states. Plan:
# reports/2026-10-22_plan_small-update-reruns.md -> E23.
#   cd /mnt/nw/home/e.agnew/commute
#   sbatch --export=ALL,STAGE=revert,SHARD=0 cluster/cancellation_sgd.sbatch   # W0 + the control FIRST, ~26 min
#   then STAGE=revert,SHARD=1 (~20 min) and STAGE=budget,SHARD=0|1 (~25 min each; refuse to start until
#   revert_shard0.json shows the control passed)
# Before: `cluster/sync.sh up`, results/predictions.json filed. Inputs on NFS: E21's states under
# experiments/{sgd_ladder_long,sgd_kv}/results/large/, experiments/transport_test/results/large/alive.pt, artifacts/.
#SBATCH --job-name=commute-cancel-sgd
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/nw/home/e.agnew/commute/cluster/logs/%x-%j.out

set -euo pipefail
source /mnt/nw/home/e.agnew/commute/cluster/env.sh
cd "$COMMUTE_REPO"
: "${STAGE:?set STAGE=revert|budget via --export}"
: "${SHARD:?set SHARD=0|1 via --export}"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$STAGE shard=$SHARD" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=experiments/cancellation_sgd/results
test -f $R/predictions.json || { echo "file the predictions first (run.py --predictions)"; exit 1; }
.venv/bin/python experiments/cancellation_sgd/run.py "$STAGE" --shard "$SHARD"
