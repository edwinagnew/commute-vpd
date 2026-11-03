#!/bin/bash
# E70: the existing/new definition read in full on the counting model. Plan:
# reports/plans/2026-11-03_plan_e70-existing-new-counting.md. Rules: experiments/pcfg_anchors/results/predictions_e70.json
# (e70.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch --job-name=commute-e70 --dependency=singleton cluster/e70.sbatch
# On exit 3 (deadline: saved, not finished) resubmit the same command by hand; compute nodes have no sbatch, so a job
# cannot resubmit itself. It resumes from results/e70.json, results/large/a1/pt_LOW.pt (a partial pretraining resumes at
# its last save), results/large/e70/probes.pt, each run's <key>_train.pt and its saved states. One job at a time (one job
# name and --dependency=singleton): every job writes e70.json and a run's _train.pt.
# Stages in order, cheapest control first: LOW pretrained (50K steps; count_a against a1_LOW.json; a miss stops), both
# probes by block, LOW's control and R1 (no teeth stops, before any fine-tune), the 20 fine-tunes, the 40 states' reads,
# R2. A stop writes results/e70_stop.json and exits 1; every later job refuses while it exists. A changed filed constant
# exits 2. Every scored number is printed to this log.
# Writes results/e70.json (under ~1 MB); states in results/large/{a1,a2,c1b}/ (12 MB each, 40 states) and
# results/large/e70/ (probes, log); results/large is a gitignored symlink into ~/commute-data/results-large/pcfg_anchors.
# Never writes a1_LOW.json or c1b.json. Commits nothing.
# Expected (estimated, not yet measured): LOW ~10-12 min, probes ~2, control ~1, fine-tunes ~2-2.5 each (~45 total),
# states ~30-45 s each (~25 total): ~85 min in all = 4-5 jobs of ~22-25 min. Per-stage seconds are in the log.
#SBATCH --job-name=commute-e70
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
.venv/bin/python experiments/pcfg_anchors/e70.py --deadline-min 22 2>&1 | tee -a "$out/e70.log" || rc=$?
if [ "$rc" = 3 ]; then
  echo "=== DEADLINE (exit 3): saved, not finished; resume by hand: sbatch --job-name=$SLURM_JOB_NAME --dependency=singleton cluster/e70.sbatch ==="
  exit 3
fi
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/ ==="
exit $rc
