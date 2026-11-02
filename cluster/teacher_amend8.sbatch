#!/bin/bash
# E34 Amendment 8: what training does with only the base model to learn from. Plan:
# reports/plans/2026-11-02_plan_base-control.md. One stage per job, each its own results file; n1 and n2 first, alone.
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_amend8.sbatch n1        # ~20 min on an L40 (estimate)
#   sbatch cluster/teacher_amend8.sbatch n2        # ~22 min
#   sbatch cluster/teacher_amend8.sbatch n3a       # ~11 min
#   sbatch cluster/teacher_amend8.sbatch n3b       # ~27 min; after n3a and the dated note of its expected values
#   sbatch cluster/teacher_amend8.sbatch n4_adam   # ~20 min
#   sbatch cluster/teacher_amend8.sbatch n4_sgd    # ~17 min
#   sbatch cluster/teacher_amend8.sbatch n5        # ~14 min
# Before: results/predictions_amend8.json filed. Inputs on NFS: transport_test/results/large/{alive,windows}.pt,
# kv_task/results/large/windows_rand32.pt, artifacts/sets/{train_0_4096,val_0_256}.npy.
# A rerun skips finished reads, on the same card and W0 floor only. Exit 1: a stop; the reason is `.stopped` in the file.
#SBATCH --job-name=commute-teacher-amend8
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 1 ] || { echo "teacher_amend8.sbatch: give exactly one stage" >&2; exit 1; }
LOGS=experiments/teacher_training/results/large/amend8
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/teacher_training/amend8.py "$1" 2>&1 | tee -a "$LOGS/run_$1.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/teacher_training/results/teacher_training_amend8_$1.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
