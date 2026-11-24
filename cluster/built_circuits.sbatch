#!/bin/bash
# E301 built_circuits (P1.63). The cluster's cards are all L40: the readings are card-dependent (SVD signs), so every
# job of one featurization reads on L40s (a pod used for a featurization must be L40/L40S for all of it).
# Usage: sbatch --comment='E301 <what>' cluster/built_circuits.sbatch <tag> <run.py arguments...>
#   sbatch --comment='E301 stage0' cluster/built_circuits.sbatch s0 stage0
#   sbatch --comment='E301 build induction' cluster/built_circuits.sbatch b0 build --circuit induction
#   sbatch --comment='E301 read vpd induction' cluster/built_circuits.sbatch v0 read --feat vpd --circuit induction --shard 0 2
# Rehearsal at full read sizes: stage0, build both circuits, then per featurization `read ... --items x2` (the controls
# run first in every read job): times one scan per featurization, peak memory, the fit time.
#SBATCH --job-name=commute-built-circuits
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
test -f experiments/built_circuits/results/predictions.json || { echo "no predictions"; exit 1; }
TAG="$1"; shift
LOG=experiments/built_circuits/results/run_${TAG}.log
( .venv/bin/python experiments/built_circuits/run.py "$@" --tag "$TAG"; s=$?; echo "=== FINISHED (exit $s) — results in $COMMUTE_REPO/experiments/built_circuits/results ==="; exit $s ) 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"  # run.py's exit, not tee's, so --dependency=afterok holds after a stop
