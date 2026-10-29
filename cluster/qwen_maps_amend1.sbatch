#!/bin/bash
# E44 Amendment 1, the layer-cut check ($0) on one cluster L40, debug QoS: experiments/qwen_maps/amend1.py. Plan:
# reports/plans/2026-10-28_plan_qwen-q0q1.md, "Amendment record (2026-10-29, Amendment 1: layer-cut check)".
# Submit from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_maps_amend1.sbatch
# δW: the cluster regen copy (default --e40, sha256 against results/regen_SHA256SUMS). Its W&B copy instead:
#   python -m commute.qwen_e40_files fetch DIR v0, then  sbatch cluster/qwen_maps_amend1.sbatch --e40 DIR
# Outputs: experiments/qwen_maps/results/amend1/ (amend1_controls.json, amend1_stage2.json, amend1_score.json,
# run.log; stopped.json on a stop rule, exit 1; a rerun refuses while it exists). Expected ~10 min, GPU ~42 GiB.
# Host: W0 and FT weights pinned (11.2 GB) + 16 rows' Γ inputs and FT logits (15.7 GB) + load transients: ~35 GB;
# 64G as the plan asks (if debug QoS refuses 64G, 48G should hold).
#SBATCH --job-name=commute-qwen_maps_amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_maps/amend1.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
