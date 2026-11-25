#!/bin/bash
# E304 built_new (P1.63 run 2): the L40 cross-check of the L40S pod reads (rule: experiments/built_new/results/l40_crosscheck_rule.md).
# Needs results/build.json and results/large/s<seed>/ home from the pod. Writes under results/<out>/ (--out), so the
# pod's own items are never skipped or overwritten.
# Usage: sbatch --comment='E304 stage: L40 cross-check' cluster/built_new.sbatch <tag> read --feat vpd --items s186001/new_few --out l40check
#SBATCH --job-name=commute-built-new
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/built_new/results/build.json || { echo "no build.json: bring the pod's build home first"; exit 1; }
TAG="$1"; shift
LOG=experiments/built_new/results/run_${TAG}.log
( .venv/bin/python experiments/built_new/run.py "$@" --tag "$TAG"; s=$?; echo "=== FINISHED (exit $s) — results in $COMMUTE_REPO/experiments/built_new/results ==="; exit $s ) 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
