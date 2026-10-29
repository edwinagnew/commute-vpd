#!/bin/bash
# E55 Stage A: PCFG minGPT pretraining. Plan: reports/plans/2026-10-29_plan_reuse-stage1.md (Stage A).
#   cd ~/commute-agents/<role>
#   sbatch cluster/pcfg_anchors.sbatch a0                                   # rehearsal, ~5-10 min
#   sbatch --job-name=commute-e55-a1-LOW cluster/pcfg_anchors.sbatch a1 LOW  # one job per prior: LOW, HIGH, NEVER
# a0: per prior, 500 pretraining iterations stopped at 250 and resumed from the saved file, behaviour and floors on 1K
#   fresh samples -> experiments/pcfg_anchors/results/a0.json; checkpoints and a0.log in results/large/a0/.
# a1 PRIOR: 50K iterations (~10 min plus compile at the a0 rate) with a 25-min deadline. On exit 3 (deadline: saved,
#   not finished) the job resubmits itself from $SLURM_SUBMIT_DIR, at most 3 times (count in E55_RESUBMITS); the next
#   job resumes from results/large/a1/pt_PRIOR.pt. Finished: results/a1_PRIOR.json; log in results/large/a1/a1_PRIOR.log.
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
usage="usage: sbatch cluster/pcfg_anchors.sbatch a0 | a1 {LOW,HIGH,NEVER}"
stage=${1:-}
prior=${2:-}
case "$stage $prior" in
  "a0 ") ;;
  "a1 LOW" | "a1 HIGH" | "a1 NEVER") ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
resubmits=${E55_RESUBMITS:-0}
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$* resubmits=$resubmits" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
out=experiments/pcfg_anchors/results/large/$stage
mkdir -p "$out"
rc=0
if [ "$stage" = a0 ]; then
  .venv/bin/python experiments/pcfg_anchors/run.py a0 2>&1 | tee "$out/a0.log" || rc=$?
  echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/a0.json ==="
  exit $rc
fi

.venv/bin/python experiments/pcfg_anchors/run.py a1 --prior "$prior" --deadline-min 25 2>&1 \
  | tee -a "$out/a1_$prior.log" || rc=$?
if [ "$rc" = 3 ]; then
  if [ "$resubmits" -ge 3 ]; then
    echo "=== DEADLINE (exit 3) after $resubmits resubmits: not resubmitting; resume by hand ==="
    exit 3
  fi
  cd "$SLURM_SUBMIT_DIR"
  E55_RESUBMITS=$((resubmits + 1)) sbatch --job-name="$SLURM_JOB_NAME" "$0" a1 "$prior"
  echo "=== DEADLINE (exit 3): resubmitted ($((resubmits + 1)) of 3) ==="
  exit 0
fi
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/a1_$prior.json ==="
exit $rc
