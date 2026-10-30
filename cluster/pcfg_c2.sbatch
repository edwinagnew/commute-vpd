#!/bin/bash
# E55 Stage C2: VPD on the C1b truth set. Plan: reports/plans/2026-10-31_plan_e55-c2-vpd.md.
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-e55-c2-rehearse cluster/pcfg_c2.sbatch rehearse     # $0 rehearsal, one job (estimate in the plan)
#   sbatch --job-name=commute-e55-c2 --dependency=singleton cluster/pcfg_c2.sbatch decompose   # after predictions_c2.json is filed
# rehearse: experiments/pcfg_anchors/c2.py rehearse: the full config's shapes at 400 steps; run A stops at 200, is
#   rebuilt from its file and resumes; run B is unbroken; then both readings on the zero update and two calibration
#   states -> results/c2_rehearsal.json; files in results/large/c2_rehearsal/ (the GB-sized training files are deleted
#   at the end, A_final.pt kept). No deadline: it is one pass; the job's 30-min limit is the stop.
# decompose: 22-min deadline, exit 3 = saved, not finished: resubmit the same command by hand (compute nodes have no
#   sbatch); it resumes from results/large/c2/decomp_train.pt. Exit 2 while predictions_c2.json is a draft.
#SBATCH --job-name=commute-e55-c2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; puts this clone's src first on PYTHONPATH
cd "$COMMUTE_REPO"
stage=${1:-}
case "$stage" in
  rehearse | decompose) ;;
  *) echo "usage: sbatch cluster/pcfg_c2.sbatch {rehearse,decompose}" >&2; exit 2 ;;
esac
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
out=experiments/pcfg_anchors/results/large/c2
mkdir -p "$out"
rc=0
.venv/bin/python experiments/pcfg_anchors/c2.py "$stage" --deadline-min 22 2>&1 | tee -a "$out/c2_$stage.log" || rc=$?
if [ "$rc" = 3 ]; then
  echo "=== DEADLINE (exit 3): saved, not finished; resume by hand: sbatch --job-name=$SLURM_JOB_NAME --dependency=singleton cluster/pcfg_c2.sbatch $stage ==="
  exit 3
fi
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/ ==="
exit $rc
