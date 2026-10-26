#!/bin/bash
# E34 Amendment 5: why the harness's and the budget sweep's adversaries disagree on the r1 SGD teacher, and whether two
# batch orders give the same copy. Plan: reports/plans/2026-10-23_plan_teacher-training.md -> "Amendment 5".
#   cd ~/commute-agents/<role>
#   sbatch cluster/teacher_amend5.sbatch     # ~27 min on an H100 (estimate); an L40 is about twice as slow
# Before: results/predictions_amend5.json filed (amend5.py --predictions); the regenerated teacher needs E34's inputs
# on NFS (experiments/transport_test/results/large/{alive,windows}.pt, experiments/task_ladder/results/large/windows_r1_t534_v369.pt).
# Card: C1 and G1 hold H100 references (W0 floor 0.075 within 5%; E34's P, size and harness ratio; A4's budget ratio).
# The L40's W0 floor has read about 0.21 (faith_score.verdict's docstring), so on this cluster C1 is expected to stop
# the job within its first minutes; the scored run belongs on a RunPod H100. A rerun resumes from
# results/teacher_training_amend5.json only on the same card.
#SBATCH --job-name=commute-teacher-amend5
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/teacher_training/results/predictions_amend5.json || { echo "file the predictions first (amend5.py --predictions)"; exit 1; }
.venv/bin/python experiments/teacher_training/amend5.py "$@"
