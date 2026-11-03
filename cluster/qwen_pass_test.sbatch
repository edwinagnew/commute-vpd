#!/bin/bash
# E68, the D1 pass test on the saved updates ($0) on one cluster L40, normal QoS (Qwen-line exception: submit only when
# the queue is empty): experiments/qwen_pass_test/run.py. Plan: reports/plans/2026-11-03_plan_e68-d1-pass-test.md. Submit
# from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_pass_test.sbatch
# Stage S's files (~32 GB per update, removed after that update's reads) go to /ephemeral/$USER/e68 on the node (--out);
# the job stops if under 40 GB is free there before S.
# Outputs: experiments/qwen_pass_test/results/ (u1..u4/<read>.json, score.json, run.log; stopped.json on a stop rule, exit 1;
# a rerun refuses while it exists). Past --wall-min (5 h 40 min, between reads) it exits 0: resubmit, it resumes from the
# read files (stage S's files are per node: on another node S runs again). Expected ~3.5 h; GPU ~46 GB (decimal); host as
# E44 Amendment 2, peak 74 GB: 80G.
#SBATCH --job-name=commute-qwen_pass_test
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# E44 Amendment 2's reader and stage S; -P keeps this script's folder off sys.path, so amend2.py's `run` is qwen_maps/run.py
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_maps:$COMMUTE_REPO/experiments/qwen_pass_test:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /ephemeral
set +e
.venv/bin/python -P experiments/qwen_pass_test/run.py --decomp "$COMMUTE_QWEN_DIR" --out "/ephemeral/$USER/e68" --wall-min 340 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
