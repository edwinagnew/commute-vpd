#!/bin/bash
# E242 (cell P1.46): R under wording A, $0 on the cluster's debug QoS. Usage: sbatch ... kstar_vpd_wording_a.sbatch TAG ARGS
# where ARGS go to run.py (--group control|built|pairs|e62|keys, --items ..., --shard I N). Every job reads the
# do-nothing first and stops on a miss; a group refuses to start before the groups it waits on are read (run.py
# docstring); an item whose JSON exists is skipped. Submissions, in order (plan reports/plans/2026-11-20_plan_r-under-a.md):
#   1. control0 --group control --items random_size_241001; control1 --group control --items random_kl_241002
#   2. built (after both, --dependency=afterok:<c0>:<c1>): built0 --items new_thin; built1 --items new_few;
#      built2 --items x2 x10; built3 --items moved rebuilt (each --group built)
#   3. once 2 has passed: pairs0..pairs25 --group pairs --shard i 26; e62_0, e62_1 --group e62 --shard i 2;
#      keys0..keys3 --group keys --shard i 4
# PRED_SHA below is git show HEAD:experiments/kstar_vpd_wording_a/results/predictions.json | sha256sum (refiled 2026-11-20 ~22:00, P2k added).
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
( PRED_SHA=2e8d08ecf801880e0e7bba7470f035960f5d582fea197730c3c3ee890995ec92 .venv/bin/python experiments/kstar_vpd_wording_a/run.py --tag "$tag" "$@" ) 2>&1 \
    | tee -a "experiments/kstar_vpd_wording_a/results/run_$tag.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E242 $tag (exit $rc) — results in $COMMUTE_REPO/experiments/kstar_vpd_wording_a/results/ ==="
exit $rc
