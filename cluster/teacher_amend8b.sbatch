#!/bin/bash
# E34 Amendment 8b: the two $0 checks reports/analyses/2026-11-02_analysis_e34-amend8-base-control.md recommends. One stage
# per job, each its own results file; either order.
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_amend8b.sbatch seeds    # ~27 min on an L40 (W0 7 min, 10 adversary starts at 115 s)
#   sbatch cluster/teacher_amend8b.sbatch moved    # ~16 min (W0 7 min, 2 trainings at 105 s, 2 reads at 130 s)
# Before: results/predictions_amend8b.json filed. Inputs on NFS: transport_test/results/large/{alive,windows}.pt,
# teacher_training/results/large/amend8/{n1,n3b,n4_adam}/, results/teacher_training_amend8_{n1,n3b,n4_adam}.json.
# A rerun skips finished reads, on the same card and W0 floor only. Exit 1: a stop; the reason is `.stopped` in the file.
#SBATCH --job-name=commute-teacher-amend8b
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
[ $# -eq 1 ] || { echo "teacher_amend8b.sbatch: give exactly one stage" >&2; exit 1; }
LOGS=experiments/teacher_training/results/large/amend8
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/teacher_training/amend8b.py "$1" 2>&1 | tee -a "$LOGS/run_amend8b_$1.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/teacher_training/results/teacher_training_amend8b_$1.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
