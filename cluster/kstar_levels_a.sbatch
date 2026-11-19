#!/bin/bash
# E214 job A: kstar_levels stages 0+1 (stage 0 do-nothing control first, exits 1 on a miss). Expected ~20 min on an L40.
#   sbatch --comment=E214 cluster/kstar_levels_a.sbatch
#SBATCH --job-name=commute-e214-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( .venv/bin/python experiments/kstar_levels/run.py --levels block mlp_attn --items reuse_small reuse_large new_few new_thin ) 2>&1 | tee -a experiments/kstar_levels/results/run_s01.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E214 job A (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_levels/results/ ==="
exit $rc
