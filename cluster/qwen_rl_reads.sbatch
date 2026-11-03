#!/bin/bash
# E75, the D1 reads on RL's weight changes ($0) on one cluster L40, normal QoS (Qwen-line exception: submit only when the queue
# is empty): experiments/qwen_rl_reads/run.py. Plan: reports/plans/2026-11-03_plan_e75-rl-reads.md. Submit from the clone root;
# not submitted by the code writer:
#   sbatch cluster/qwen_rl_reads.sbatch                                # control, u1, R100, R10, u2, u3
#   sbatch cluster/qwen_rl_reads.sbatch --updates N100,N10,F100,F10    # when those land
# Stage S's files (~32 GB per update, removed after that update's reads) go to /ephemeral/$USER/e75 on the node (--out); the job
# stops if under 40 GB is free there before S.
# Outputs: experiments/qwen_rl_reads/results/ (u1/ the control, <update>/<read>.json, score.json, run.log; stopped.json on a stop
# rule, exit 1; a rerun refuses while it exists). Past --wall-min (5 h 40 min, between reads) it exits 0: resubmit, it resumes from
# the read files (stage S's files are per node: on another node S runs again). Expected: control ~5 min, u1 and u2 ~40 min each, u3 ~35,
# R100 and R10 ~1 h each, ~4 h in all; GPU ~46 GB (decimal); host as E72, peak 81 GB, plus ~3 GB for fp32 δW; Σ kept (already in that peak).
#SBATCH --job-name=commute-qwen_rl_reads
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# E68's reader and stage S; -P keeps this script's folder off sys.path, so amend2.py's `run` is qwen_maps/run.py
export PYTHONPATH=$COMMUTE_REPO/experiments/qwen_maps:$COMMUTE_REPO/experiments/qwen_pass_test:$PYTHONPATH
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /ephemeral
set +e
.venv/bin/python -P experiments/qwen_rl_reads/run.py --decomp "$COMMUTE_QWEN_DIR" --out "/ephemeral/$USER/e75" --wall-min 340 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
