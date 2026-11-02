#!/bin/bash
# E34 Amendment 8c: asked for by reports/reviews/2026-11-02_review_amend8-analysis-check.md, items 1, 2, 5. A null student at
# batch order 1 and cross-teacher cosines in delta-W. One job, one results file.
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_amend8c.sbatch         # ~16 min on an L40 (W0 7 min, T1000 30 s, 4 students at 105 s, saves)
# Before: results/predictions_amend8c.json filed. Inputs on NFS: transport_test/results/large/{alive,windows}.pt,
# results/teacher_training_amend8_n3b.json. Writes results/large/amend8/amend8c/ (6 files of 60 MB, uv0.pt).
# A rerun retrains (deterministic), on the same card and W0 floor only. Exit 1: a stop; the reason is `.stopped` in the file.
#SBATCH --job-name=commute-teacher-amend8c
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
[ $# -eq 0 ] || { echo "teacher_amend8c.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/teacher_training/results/large/amend8
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/teacher_training/amend8c.py 2>&1 | tee -a "$LOGS/run_amend8c.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/teacher_training/results/teacher_training_amend8c.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
