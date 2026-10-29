#!/bin/bash
# E55 Stage A0: rehearsal of the PCFG minGPT pretraining at tiny sizes (~5-10 min). Plan:
# reports/plans/2026-10-29_plan_reuse-stage1.md (Stage A).
#   cd ~/commute-agents/<role>
#   sbatch cluster/pcfg_anchors.sbatch a0
# Per prior (LOW, HIGH, NEVER): 500 pretraining iterations stopped at 250 and resumed from the saved file, count-a and
# count-b accuracy on 1K fresh samples. Writes experiments/pcfg_anchors/results/a0.json (ms/step, generation rate,
# peak GPU memory, checkpoint size) and results/large/a0/ (checkpoints, a0.log). Not a reading: no predictions needed.
#SBATCH --job-name=commute-e55-a0
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; puts this clone's src first on PYTHONPATH
cd "$COMMUTE_REPO"
[ "${1:-}" = a0 ] || { echo "usage: sbatch cluster/pcfg_anchors.sbatch a0" >&2; exit 2; }
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p experiments/pcfg_anchors/results/large/a0
rc=0
.venv/bin/python experiments/pcfg_anchors/run.py a0 2>&1 | tee experiments/pcfg_anchors/results/large/a0/a0.log || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/a0.json ==="
exit $rc
