#!/bin/bash
# E132 (cell P3.1): the re-gating share on the new decomposition p-9707d6ff (experiments/qwen_regating_new/run.py; plan
# reports/plans/2026-11-09_plan_e132-regating-new.md). One L40, debug QoS, $0. Submit from the clone root; not by hand,
# through a pod-runner: sbatch cluster/qwen_regating_new.sbatch
# Expected, from E57's job (same card, old decomposition, results/run.log): loads ~3 min; controls ~5 min; Σ ~3 min;
# per state split ~6 min (no nulls) + two cell passes ~4 min; six states ~65 min; ~80 min in all. GPU ~42.6 GiB in the
# cells (stop at 44 after the first row; Γ parked during Σ and the split). Host ~93 GB (E57 C0 with the Gaussian held).
#SBATCH --job-name=commute-qwen_regating_new
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot is here already
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_regating_new/run.py --decomp "$COMMUTE_QWEN_NEW_DIR"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
