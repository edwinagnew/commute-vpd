#!/bin/bash
# E42: teacher training on a behaviour the model lacks (key -> value cipher). Plan:
# reports/plans/2026-10-28_plan_e42-teacher-new-behaviour.md; brief experiments/teacher_training/RUN_BRIEF_e42.md.
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_e42.sbatch rand32            # job 1: ~30 min on an L40 (estimate)
#   sbatch cluster/teacher_e42.sbatch ident32           # jobs 2, 3 per the plan's decision tree, in parallel
#   sbatch cluster/teacher_e42.sbatch rand1
# Before: results/predictions_e42.json filed (newskill.py --predictions). Inputs on NFS: kv_task/results/large/windows_*.pt,
# transport_test/results/large/{alive,windows}.pt. One results file per task (teacher_training_e42_<task>.json); a rerun
# resumes it only on the same card with the same W0 floor.
#SBATCH --job-name=commute-teacher-e42
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/teacher_training/results/predictions_e42.json || { echo "file the predictions first (newskill.py --predictions)"; exit 1; }
.venv/bin/python experiments/teacher_training/newskill.py "$@"
