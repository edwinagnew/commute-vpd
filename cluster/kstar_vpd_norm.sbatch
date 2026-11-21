#!/bin/bash
# E226 Amendment 1 (P1.37 run 2): size-normalised R on E226's saved K* and s, every item, $0 on the cluster.
# Stage 0 (construction checks, do-nothing) first; stops on a miss. A rerun skips items whose JSON exists.
#   sbatch --comment='E226 rerun: size-normalised R' cluster/kstar_vpd_norm.sbatch
#SBATCH --job-name=commute-e226-norm
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=5d313ac5cdc94918a5c5cd58c2f285a939b8ee48c77e4d599298cc5fc4b0e1b8 .venv/bin/python experiments/kstar_vpd_norm/run.py ) 2>&1 | tee -a experiments/kstar_vpd_norm/results/run.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E226 Amendment 1 (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_vpd_norm/results/ ==="
exit $rc
