#!/bin/bash
# E70 Amendment 1: remove the direction the count moves the residual along. Plan:
# reports/plans/2026-11-03_plan_e70-existing-new-counting.md, "Amendment 1". Rules:
# experiments/pcfg_anchors/results/predictions_e70_a1.json (e70_a1.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch --job-name=commute-e70-a1 --dependency=singleton cluster/e70_a1.sbatch
# On exit 3 (deadline: saved, not finished) resubmit the same command by hand; it resumes from results/e70_a1.json. One job
# at a time (one job name and --dependency=singleton). A stop writes results/e70_a1_stop.json and exits 1, and every later
# job refuses while it exists. A changed filed constant exits 2. A missing saved file exits 1 with its name.
# Reads, trains nothing: results/large/a1/pt_LOW.pt, results/large/e70/probes.pt, results/e70.json, and the 20 states at
# 10,000 steps in results/large/{a2,c1b}/. Writes results/e70_a1.json (under ~1 MB) and results/large/e70/e70_a1.log.
# Expected, scaled from E70's log (control 82 s for 250 passes; a state 81 s for 250 passes), not measured: control ~1 min,
# why ~1 min, LOW on the state rows ~1 min, states ~40 s each (~14 min), verdict 0: ~17 min. The stop leaves after ~2 min.
#SBATCH --job-name=commute-e70-a1
#SBATCH --dependency=singleton
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
out=experiments/pcfg_anchors/results/large/e70
mkdir -p "$out"
rc=0
.venv/bin/python experiments/pcfg_anchors/e70_a1.py --deadline-min 22 2>&1 | tee -a "$out/e70_a1.log" || rc=$?
if [ "$rc" = 3 ]; then
  echo "=== DEADLINE (exit 3): saved, not finished; resume by hand: sbatch --job-name=$SLURM_JOB_NAME --dependency=singleton cluster/e70_a1.sbatch ==="
  exit 3
fi
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/ ==="
exit $rc
