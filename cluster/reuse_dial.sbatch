#!/bin/bash
# P2.63 the reuse dial: the $0 rehearsal on one L40 (debug QoS). Plan reports/plans/2026-11-24_plan_p2-63-reuse-dial.md
# (appendix, gate 3). Rules: experiments/reuse_dial/results/predictions.json (run.py refuses unless committed unchanged).
#   cd ~/commute-agents/compare-lead && sbatch --comment='E300 rehearse: reuse dial, calibrate procedural' cluster/reuse_dial.sbatch calibrate addition sequence brackets
#   cd ~/commute-agents/compare-lead && sbatch --comment='E300 rehearse: reuse dial, calibrate lowercase' cluster/reuse_dial.sbatch calibrate lowercase
#   cd ~/commute-agents/compare-lead && sbatch --comment='E300 rehearse: reuse dial, item' cluster/reuse_dial.sbatch rehearse
# calibrate: stage 0, then for each of the four tasks the gains-only seed-0 arm to its cap and every rehearse beta, beta 0
#   and the ordinary arm to min(t_ref, 0.9 x plateau): results/rehearse/calibrate_<task>.json (procedural ~30 min for
#   three tasks; lowercase up to ~60 min: 3,000 gains-only steps with a read-row check each step). Log per first task.
# rehearse: stage 0, then item (addition, seed 0) at rehearse_cap steps per arm, every state read in full, the full E_b
#   scan on arms g and ft; then the same command again, which must skip every part (resume) (~70 min).
# Exit 1 at a stage-0 miss or a filed early check (the log's STOP line says which). Writes only under results/.
#SBATCH --job-name=commute-p2-63-reuse-dial
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ -e experiments/reuse_dial/results/large ] || "$HOME/commute-agents/fleet/link-data" "$COMMUTE_REPO"
mkdir -p experiments/reuse_dial/results/rehearse
log=experiments/reuse_dial/results/rehearse/$(echo "$*" | tr ' ' '_').log
rc=0
if [ "$1" = calibrate ]; then
  shift
  [ $# -gt 0 ] || set -- addition sequence brackets lowercase
  .venv/bin/python experiments/reuse_dial/run.py calibrate "$@" --tag "cal_$1" 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
else
  .venv/bin/python experiments/reuse_dial/run.py rehearse --tag r0 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then
    echo "=== resume check: the same command again; every part must be skipped ===" | tee -a "$log"
    .venv/bin/python experiments/reuse_dial/run.py rehearse --tag r0 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
  fi
fi
echo "=== FINISHED P2.63 $* (exit $rc) — results in $COMMUTE_REPO/experiments/reuse_dial/results/rehearse/ ==="
exit $rc
