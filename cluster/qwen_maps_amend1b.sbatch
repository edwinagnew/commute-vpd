#!/bin/bash
# E44 Amendment 1b, consistent inputs ($0) on one cluster L40, debug QoS: experiments/qwen_maps/amend1b.py. Plan:
# reports/plans/2026-10-28_plan_qwen-q0q1.md, "Amendment record (2026-10-29, Amendment 1b: consistent inputs)".
# Submit from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_maps_amend1b.sbatch
# δW: the cluster regen copy (default --e40, sha256 against results/regen_SHA256SUMS), as Amendment 1.
# Outputs: experiments/qwen_maps/results/amend1b/ (amend1b_controls.json, amend1b_stage2.json, amend1b_score.json,
# run.log; stopped.json on a stop rule, exit 1; a rerun refuses while it exists). Expected ~7 min, GPU ~42 GiB.
# Host: Amendment 1's log read a 52.5 GB peak resident set at --mem=48G (job 44065), so 64G here.
#SBATCH --job-name=commute-qwen_maps_amend1b
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
.venv/bin/python experiments/qwen_maps/amend1b.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
