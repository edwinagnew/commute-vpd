#!/bin/bash
# E44 Amendment 2, the pass test under reading B ($0) on one cluster L40, normal QoS (Qwen-line exception: submit
# only when the queue is empty, else debug with step 1 cut): experiments/qwen_maps/amend2.py. Plan:
# reports/plans/2026-10-28_plan_qwen-q0q1.md, "Amendment record (2026-10-30, Amendment 2: masks from the original
# model)". Submit from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_maps_amend2.sbatch
# δW: the cluster regen copy (default --e40, sha256 against results/regen_SHA256SUMS). Stage S's files (~64 GB) go to
# /ephemeral/$USER/e44a2 on the node (--out); the job stops if under 80 GB is free there before S.
# Outputs: experiments/qwen_maps/results/amend2/ (controls, S, t10, m10, t1, m1 .json, score.json, run.log;
# stopped.json on a stop rule, exit 1; a rerun refuses while it exists). Cut by the 2 h 45 min wall rule (exit 0) or
# the time limit: resubmit, it resumes from the stage files. Expected ~1 h 45 min; GPU ~42.5 GiB in reads, ~36 in S.
# Host: E44's A100 host peak 59.2 GB with Σ and pinned weights; here three pinned weight sets (16.8 GB): 80G.
#SBATCH --job-name=commute-qwen_maps_amend2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /ephemeral
set +e
.venv/bin/python experiments/qwen_maps/amend2.py --decomp "$COMMUTE_QWEN_DIR" --out "/ephemeral/$USER/e44a2" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
