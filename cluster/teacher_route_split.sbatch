#!/bin/bash
# E119: the route split of E34's teacher-trained decompositions on the trigger tasks. Plan reports/plans/2026-11-06_plan_e119-teacher-route-split-triggers.md.
#   cd ~/commute-agents/<role>                         # a clone with the results/large symlinks (fleet/link-data <role> after pulling), not a worktree
#   sbatch cluster/teacher_route_split.sbatch TASK     # one of r1_t534_v369 r0_t352_v310 emoticon; ~13 min each on an L40 (plan, Timing)
#   sbatch cluster/teacher_route_split.sbatch          # all three in one job: ~36 min, over the 30-min limit; use one job per task
# Before: results/predictions.json committed. Inputs on NFS: transport_test/results/large/{alive,windows}.pt,
# task_ladder/results/large/windows_{r0_t352_v310,r1_t534_v369}.pt, artifacts/sets/train_32768_8192.npy (arm B's Pile rows).
# Writes results/teacher_route_split_<task>.json, results/large/<task>/{sgd,adam}_{A,B}.pt (4 x ~60 MB per task), and
# results/score.json when the third task finishes. A rerun skips every student with a record and an offsets file.
# Exit 1: a stop; the reason is `.stopped` in the task's file.
#SBATCH --job-name=commute-e119
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
LOGS=experiments/teacher_route_split/results/large
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/teacher_route_split/run.py "$@" 2>&1 | tee -a "$LOGS/run_${1:-all}.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/teacher_route_split/results/ and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
