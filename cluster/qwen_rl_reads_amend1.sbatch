#!/bin/bash
# E75 Amendment 1, three bad placements of the right size ($0) on one cluster L40, debug QoS: experiments/qwen_rl_reads/amend1.py.
# Plan: reports/plans/2026-11-03_plan_e75-rl-reads.md, "Amendment 1". Submit from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_rl_reads_amend1.sbatch
# Outputs: experiments/qwen_rl_reads/results/ (amend1/<update>/<read>.json, amend1/run.log, amend1_score.json; stopped_amend1.json on
# a miss of the control, exit 1; a rerun refuses while it exists). Touches nothing E75's run.py writes. Past --wall-min (40 min,
# between reads) it exits 0: resubmit, it resumes from the read files; `sbatch ... --wall-min N` overrides the limit.
# Expected ~25 min (17 reads at 70 s, u3's 35 s). No masks, so no Γ or components: GPU ~10 GB, host ~30 GB. No stage S, nothing on /ephemeral.
#SBATCH --job-name=commute-qwen_rl_reads_amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
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
.venv/bin/python -P experiments/qwen_rl_reads/amend1.py --wall-min 40 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
