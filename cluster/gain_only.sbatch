#!/bin/bash
# P2.46 gain-only training: the $0 rehearsal on one L40 (debug QoS), or one shard of the scored run.
# Plan: reports/plans/2026-11-20_plan_p2-46-gain-only.md. Rules: experiments/gain_only/results/predictions.json (run.py
# refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && sbatch --comment='E### rehearse: gain-only, four recipe families at full read sizes' cluster/gain_only.sbatch rehearse
# rehearse: stage 0, then four items (one per recipe family) capped at 20 steps, trained, exported and read at full read
# sizes into experiments/gain_only/results/rehearse/; then the same command again, which must skip every item (resume).
# Exit 1 at a stage-0 miss or a step-1 KL overshoot (the log's STOP line says which). Writes only under results/.
# Time: expected 10-20 min, mostly E228's split bases (once) and the four reads; the limit is 30 min.
#SBATCH --job-name=commute-p2-46-gain-only
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
large=experiments/gain_only/results/large
[ -e "$large" ] || "$HOME/commute-agents/fleet/link-data" "$COMMUTE_REPO"
mkdir -p experiments/gain_only/results/rehearse
log=experiments/gain_only/results/${1:-run}.log
rc=0
.venv/bin/python experiments/gain_only/run.py "$@" 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ] && [ "${1:-}" = rehearse ]; then
  echo "=== resume check: the same command again; every item must be skipped ===" | tee -a "$log"
  .venv/bin/python experiments/gain_only/run.py "$@" 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
fi
echo "=== FINISHED P2.46 $* (exit $rc) — results in $COMMUTE_REPO/experiments/gain_only/results/ ==="
exit $rc
