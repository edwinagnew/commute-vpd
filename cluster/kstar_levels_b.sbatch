#!/bin/bash
# E214 job B: kstar_levels stage 2. Expected 45-60 min on an L40.
#   sbatch --comment="E214 stage: 2" cluster/kstar_levels_b.sbatch
#SBATCH --job-name=commute-e214-b
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( .venv/bin/python experiments/kstar_levels/run.py --levels block mlp_attn head_mlp ) 2>&1 | tee -a experiments/kstar_levels/results/run_s2.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E214 job B (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_levels/results/ ==="
exit $rc
