#!/bin/bash
# E263 (cell P1.52): E_b at E234's SAE latents on the saved E245/E251 items, $0 on the cluster's debug QoS. Usage:
#   sbatch --comment='E263 <tag>' cluster/sae_eb.sbatch TAG --shard I N
# Every job reads the do-nothing first and stops on a miss; an item whose JSON exists is skipped (run.py docstring).
# PRED_SHA below is git show HEAD:experiments/sae_eb/results/predictions.json | sha256sum.
#SBATCH --job-name=commute-sae-eb
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
tag=${1:?tag, e.g. eb0}
shift
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
( PRED_SHA=d91c6d02a0d58bde45819bc1d16c0376de1a4595164993c2eea538fd40a6222d .venv/bin/python experiments/sae_eb/run.py --tag "$tag" "$@" ) 2>&1 \
    | tee -a "experiments/sae_eb/results/run_$tag.log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED E263 $tag (exit $rc) — results in $COMMUTE_REPO/experiments/sae_eb/results/ ==="
exit $rc
