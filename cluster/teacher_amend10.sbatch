#!/bin/bash
# E34 Amendment 10: the student's change against the fine-tune's, weighted by the inputs the model sees. Plan reports/plans/2026-11-02_plan_e34-amend10-input-weighted.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks and artifacts/sets, not a worktree
#   sbatch cluster/teacher_amend10.sbatch           # ~12 min on an L40 (model load 23 s as in E12 Amendment 3, two 1,000-step fine-tunes, three window sets, 256 Pile rows, eigendecompositions)
# Before: results/predictions_amend10.json and results/amend9.json present. Inputs on NFS: teacher_training/results/large/amend8/amend8c/, new_vs_existing/results/large/amend2/,
# regating_table/results/large/{rand32,rand32b}_s{0,1}.pt and windows_*.pt, transport_test/results/large/windows.pt, artifacts/sets/val_0_256.npy.
# Writes results/amend10.json and results/large/amend10/{real_teacher_o1_dW.pt,gram_*.pt,run_amend10.log}. A rerun refuses while the results file exists: move it aside.
# Exit 1: a stop; the reason is `.stopped` in the file.
#SBATCH --job-name=commute-e34-amend10
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
export PYTHONPATH=$COMMUTE_REPO/experiments/new_vs_existing:$PYTHONPATH   # amend9 loads E12's amend2, which imports amend1; the script's own folder stays first
[ $# -eq 0 ] || { echo "teacher_amend10.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/teacher_training/results/large/amend10
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/teacher_training/amend10.py 2>&1 | tee -a "$LOGS/run_amend10.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/teacher_training/results/amend10.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
