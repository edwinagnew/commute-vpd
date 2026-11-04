#!/bin/bash
# E86, fractions of the update and the clean table on one cluster L40, debug QoS, $0: experiments/qwen_d1_fractions/run.py. Plan:
# reports/plans/2026-11-04_plan_e86-update-fractions.md. Submit from the clone root, u1 first and alone; not submitted by the code writer:
#   sbatch cluster/qwen_d1_fractions.sbatch --update u1      # the control; the others refuse until it ends
#   sbatch cluster/qwen_d1_fractions.sbatch --update tofu
#   sbatch cluster/qwen_d1_fractions.sbatch --update u2      # the gold fine-tune
# Exit 0: done, or past --wall-min (26, between reads): if results/<update>/left_in_delta.json is absent, resubmit, it resumes (the X on /ephemeral stay).
# Exit 1: a stop (results/stopped.json; a rerun refuses while it exists), or under 45.6 GB free on the card (no stopped.json: resubmit).
# Expected about 25 min per update. GPU under 45.6 GB, no Γ. Host: E75's 61.5 GB through stage S, with Σ; --mem is that and a margin.
#SBATCH --job-name=commute-qwen_d1_fractions
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# E72's reader; -P keeps this script's folder off sys.path, so qwen_maps/amend2.py's `run` is qwen_maps/run.py
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_maps:$COMMUTE_REPO/experiments/qwen_pass_test:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
df -h /ephemeral
set +e
.venv/bin/python -P experiments/qwen_d1_fractions/run.py --decomp "$COMMUTE_QWEN_DIR" --out "/ephemeral/$USER/e86" --wall-min 26 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
