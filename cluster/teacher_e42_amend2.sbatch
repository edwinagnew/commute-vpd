#!/bin/bash
# E42 Amendment 2: each masked term alone with nothing to copy, then the cipher copy with the term(s) that hold still
# (rand32 only, one job). Plan: reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> "Amendment 2"; brief
# experiments/teacher_training/RUN_BRIEF_e42_amend2.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_e42_amend2.sbatch        # ~25-33 min on an L40 (estimate from job 40338)
# Before: results/predictions_e42_amend2.json filed (newskill_amend2.py --predictions). Inputs on NFS:
# kv_task/results/large/windows_rand32.pt, transport_test/results/large/{alive,windows}.pt,
# teacher_training/results/large/e42/rand32/A_step1000.pt and results/teacher_training_e42_rand32.json (E42's).
# A rerun resumes its results file only on the same card with the same W0 floor.
#SBATCH --job-name=commute-teacher-e42a2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 0 ] || { echo "teacher_e42_amend2.sbatch: takes no arguments (rand32 only)" >&2; exit 1; }
test -f experiments/teacher_training/results/predictions_e42_amend2.json \
  || { echo "file the predictions first (newskill_amend2.py --predictions)"; exit 1; }
rc=0
.venv/bin/python experiments/teacher_training/newskill_amend2.py || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/teacher_training/results/teacher_training_e42_amend2_rand32.json and $COMMUTE_REPO/experiments/teacher_training/results/large/e42_amend2/rand32/ ==="
exit $rc
