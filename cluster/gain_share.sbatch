#!/bin/bash
# P1.58 (E282) the gain-only share: stages 0, 1, 2 and 3 on one L40 (debug QoS), or a manifest of further states (list).
# Card experiments/gain_share/card.md; plan reports/plans/2026-11-23_plan_p1-58-gain-share.md; rules in
# experiments/gain_share/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && sbatch --comment='E282 gain share: stages 0-3' cluster/gain_share.sbatch all
#   cd ~/commute-agents/compare-lead && sbatch --comment='E282 gain share: more states' cluster/gain_share.sbatch list --manifest PATH
# Exit 1 at any stage's miss (the log's STOP line says which, the stage's JSON holds "stopped"); a rerun skips every
# (state, level) already written. Writes only under experiments/gain_share/results/.
# Time: expected 20-25 min (28 fits of 600 steps at about 0.03-0.06 s/step, the reads, the model and edits built once);
# the limit is 30 min.
#SBATCH --job-name=commute-p1-58-gain-share
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
large=experiments/gain_share/results/large
[ -e "$large" ] || "$HOME/commute-agents/fleet/link-data" "$COMMUTE_REPO"
log=experiments/gain_share/results/run_${1:-all}.log
rc=0
.venv/bin/python experiments/gain_share/run.py "$@" 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED P1.58 E282 $* (exit $rc) — results in $COMMUTE_REPO/experiments/gain_share/results/ ==="
exit $rc
