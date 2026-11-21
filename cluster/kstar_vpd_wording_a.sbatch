#!/bin/bash
# E242 (cell P1.46): R under wording A, $0 on the cluster's debug QoS. Usage: sbatch ... kstar_vpd_wording_a.sbatch TAG ARGS
# where ARGS go to run.py (--group control|built|pairs|e62|keys, --items ..., --shard I N). Every job reads the
# do-nothing first and stops on a miss; a group refuses to start before the groups it waits on are read (run.py
# docstring); an item whose JSON exists is skipped. Submissions, in order (plan reports/plans/2026-11-20_plan_r-under-a.md):
#   1. control:  sbatch --comment='E242 control' cluster/kstar_vpd_wording_a.sbatch control --group control
#   2. built (after 1, --dependency=afterok:<1>): built0 --group built --items new_thin; built1 ... new_few;
#      built2 ... x2 x10 moved rebuilt
#   3. once 2 has passed: pairs0..pairs15 --group pairs --shard i 16; e62 --group e62; keys0, keys1 --group keys --shard i 2
# PRED_SHA below is git show HEAD:experiments/kstar_vpd_wording_a/results/predictions.json | sha256sum (filed 2026-11-20).
# Afterwards: ~/commute-agents/fleet/cpu-run python3 experiments/kstar_vpd_wording_a/score.py
#SBATCH --job-name=commute-wording-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
tag=${1:?tag, e.g. control, built0 or pairs3}
shift
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=85e4974e0017a1684f3f8afe07abcbd611a261a5e472853bd4c7be553ebf6f2f .venv/bin/python experiments/kstar_vpd_wording_a/run.py --tag "$tag" "$@" ) 2>&1 \
    | tee -a "experiments/kstar_vpd_wording_a/results/run_$tag.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E242 $tag (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_vpd_wording_a/results/ ==="
exit $rc
