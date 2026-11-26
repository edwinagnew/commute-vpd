#!/bin/bash
# E311 (cell P2.83): one scored stage of experiments/crosscoder_fig4/run.py on a MATS debug L40, full size.
# Usage (from the clone root): sbatch --comment='E311 stage: <what>' cluster/crosscoder_fig4.sbatch <run.py args...>
#   e.g. `... 0a`, `... 0b random --seed 0`, `... 2 i/26`. A job that hits the time limit resumes from its last
#   checkpoint (results/large/ckpt/, every 10,000 steps) when resubmitted with the same args (fleet/resubmit).
# Plan reports/plans/2026-11-25_plan_p2-83-crosscoder-fix.md. Pods were the plan; this is the cluster fallback while
# RUNPOD_API_KEY is unset (2026-11-25).
#SBATCH --job-name=commute-crosscoder-fig4
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/crosscoder_fig4/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
T0=$(date +%s)
.venv/bin/python experiments/crosscoder_fig4/run.py "$@"
echo "FINISHED $* in $(( $(date +%s) - T0 )) s"
