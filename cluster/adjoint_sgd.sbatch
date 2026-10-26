#!/bin/bash
# adjoint_sgd (E31): the adjoint split of SGD and Adam fine-tunes, in shards on the L40 (remote-kernels was down
# 2026-10-22). Plan: reports/2026-10-22_plan_adjoint-sgd.md. Each shard runs its own same-job W0 floor first.
#   cd ~/commute-agents/<role>
#   sbatch --export=ALL,SHARD=A cluster/adjoint_sgd.sbatch    # the reproductions (gated) FIRST, ~45 min
#   then SHARD=B|C|D|E (~30-45 min each); they refuse to start until results/adjoint_sgd_A.json shows W0 and the
#   Adam 3e-5 and SGD 0.01 reproductions passed. Afterwards (laptop): run.py merge.
# Before: `cluster/sync.sh up`, results/predictions.json filed. Inputs on NFS: experiments/transport_test/results/large/,
# experiments/task_ladder/results/large/windows_*.pt, artifacts/.
#SBATCH --job-name=commute-adjoint-sgd
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:05:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
: "${SHARD:?set SHARD=A|B|C|D|E via --export}"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS shard=$SHARD" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/adjoint_sgd/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/adjoint_sgd/run.py shard "$SHARD"
