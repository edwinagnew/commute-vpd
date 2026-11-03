#!/bin/bash
# E77 Amendment 1, hook control, swap read, later blocks, then reads 2 and 3 ($0) on one cluster L40, normal QoS (Qwen-line exception:
# submit only when the queue is empty): experiments/qwen_open_points/amend1.py. Plan: reports/plans/2026-11-03_plan_e77-open-points.md.
# Submit from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_open_points_amend1.sbatch
# Outputs: experiments/qwen_open_points/results/ (u1..u3/<read>.json, score.json, run.log; stopped.json on a stop rule,
# exit 1; a rerun refuses while it exists). Past --wall-min (110 min, between reads) it exits 0: resubmit, it resumes from the read
# files. Expected ~1 h; GPU ~46 GB (decimal); host as E72 before its stage S, ~45 GB. No stage S and no files on /ephemeral.
#SBATCH --job-name=commute-qwen_open_points_amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# E68's and E72's reader; -P keeps this script's folder off sys.path, so amend2.py's `run` is qwen_maps/run.py
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_maps:$COMMUTE_REPO/experiments/qwen_pass_test:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python -P experiments/qwen_open_points/amend1.py --decomp "$COMMUTE_QWEN_DIR" --wall-min 110 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
