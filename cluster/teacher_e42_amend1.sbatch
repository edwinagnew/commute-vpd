#!/bin/bash
# E42 Amendment 1: teacher training with the masked students matched to the teacher, on the key -> value cipher. Plan:
# reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md -> "Amendment 1"; brief
# experiments/teacher_training/RUN_BRIEF_e42_amend1.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_e42_amend1.sbatch rand32     # job 1: ~21 min on an L40 (estimate)
#   sbatch cluster/teacher_e42_amend1.sbatch rand1      # job 2, after job 1 passed C1, G2, M0 and G1
# Before: results/predictions_e42_amend1.json filed (newskill_amend1.py --predictions). Inputs on NFS:
# kv_task/results/large/windows_<task>.pt, transport_test/results/large/{alive,windows}.pt,
# teacher_training/results/large/e42/<task>/A_step1000.pt and results/teacher_training_e42_<task>.json (E42's).
# One task per job; a rerun resumes its results file only on the same card with the same W0 floor.
#SBATCH --job-name=commute-teacher-e42a1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 1 ] || { echo "teacher_e42_amend1.sbatch: give exactly one task (rand32 or rand1)" >&2; exit 1; }
test -f experiments/teacher_training/results/predictions_e42_amend1.json \
  || { echo "file the predictions first (newskill_amend1.py --predictions)"; exit 1; }
rc=0
.venv/bin/python experiments/teacher_training/newskill_amend1.py "$1" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/teacher_training/results/teacher_training_e42_amend1_$1.json and $COMMUTE_REPO/experiments/teacher_training/results/large/e42_amend1/$1/ ==="
exit $rc
