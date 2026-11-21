#!/bin/bash
# E245 (cell P1.40 run 2): R under wording A at E234's SAE latents, $0 on the cluster's debug QoS. Usage:
#   sbatch --comment='E245 <tag>' cluster/sae_r_a.sbatch TAG ARGS
# where ARGS go to run.py (--group built|keys|pairs, --shard I N, --size rehearse). Every job reads the do-nothing
# first and stops on a miss; an item whose JSON exists is skipped; keys and pairs refuse to start until the built
# group's order check has held (run.py docstring). Submissions, in order (plan reports/plans/2026-11-21_plan_r-under-a-sae.md):
#   0. rehearse --group built --size rehearse          (do-nothing and x10, eval cap 50)
#   1. built0..built2 --group built --shard i 3        (the last to finish runs the order check)
#   2. once it holds: keys0..keys3 --group keys --shard i 4; then, if time allows, pairs0.. --group pairs --shard i N
# PRED_SHA below is git show HEAD:experiments/sae_r_a/results/predictions.json | sha256sum (filed 2026-11-21).
#SBATCH --job-name=commute-sae-r-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
tag=${1:?tag, e.g. rehearse, built0 or keys2}
shift
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=8faf2d193cf1f203eb292c16d4081c1bd2adbbdb04aa180a7b4320eea92b17b8 .venv/bin/python experiments/sae_r_a/run.py --tag "$tag" "$@" ) 2>&1 \
    | tee -a "experiments/sae_r_a/results/run_$tag.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E245 $tag (exit $rc) — results in $COMMUTE_REPO/experiments/sae_r_a/results/ ==="
exit $rc
