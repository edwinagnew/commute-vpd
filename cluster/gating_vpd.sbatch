#!/bin/bash
# E241 (P1.45): the causal test with gating at VPD, $0 on the cluster. Two jobs at once, any order:
#   sbatch --comment='E241 built: gating at VPD' cluster/gating_vpd.sbatch built
#   sbatch --comment='E241 pairs: gating at VPD' cluster/gating_vpd.sbatch pairs
# Each job reads its do-nothing state first (stops on a miss). A rerun skips items whose JSON exists. Then, no GPU:
#   fleet/cpu-run python3 experiments/gating_vpd/score.py
# PRED_SHA: sha256 of experiments/gating_vpd/results/predictions.json as committed in 92367454.
#SBATCH --job-name=commute-e241-gating
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
part=${1:?part: built or pairs}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=9fb48844f2a0df207f3ebb8cf9bb781c57fc51891b3b72ff27122a77f7125eea.venv/bin/python experiments/gating_vpd/run.py "$part" ) 2>&1 \
    | tee -a "experiments/gating_vpd/results/run_$part.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E241 $part (exit $rc) — results in $COMMUTE_REPO/experiments/gating_vpd/results/ ==="
exit $rc
