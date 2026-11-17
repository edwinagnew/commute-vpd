#!/bin/bash
# E144 = E132 Amendment 1 (cell P3.2): the log-odds re-gating share on the new decomposition p-9707d6ff
# (experiments/qwen_regating_new/amend1.py; plan reports/plans/2026-11-10_plan_e144-e132-amend1-logodds.md). One L40,
# debug QoS, $0. Submit from the clone root, through a pod-runner:
#   sbatch --comment=E144 cluster/qwen_regating_amend1.sbatch
# Expected, from E132's job (same card and decomposition, results/run.log): loads ~1.5 min; one pass of 64 rows of 512
# (clean forward with Γ + masked forward) ~1 min, 15 passes ~15-20 min; Σ ~3 min; KL checks and weight copies ~5 min;
# ~25-35 min in all. GPU ~43 GiB (E132 42.0 at rows of 256; stop at 44 after the first row). Host ~95 GB (E132 91).
# A resubmit skips the stages already written (results/amend1/).
#SBATCH --job-name=commute-e144-qwen_regating_amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot and tokenizer are here already
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_regating_new/amend1.py --decomp "$COMMUTE_QWEN_NEW_DIR"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
