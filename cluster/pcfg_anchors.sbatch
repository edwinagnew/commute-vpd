#!/bin/bash
# E55 Stage A: PCFG minGPT anchors. Plan: reports/plans/2026-10-29_plan_reuse-stage1.md (Stage A).
#   cd ~/commute-agents/<role>
#   sbatch cluster/pcfg_anchors.sbatch a0                                   # rehearsal, ~5-10 min
#   sbatch --job-name=commute-e55-a1-LOW cluster/pcfg_anchors.sbatch a1 LOW  # one job per prior: LOW, HIGH, NEVER
#   sbatch --job-name=commute-e55-a1c cluster/pcfg_anchors.sbatch a1c       # after all three a1 jobs
#   sbatch --job-name=commute-e55-a2-1 cluster/pcfg_anchors.sbatch a2 N1 N2 N3 W2 R   # arms in order, ~3 min each
# a0: per prior, 500 pretraining iterations stopped at 250 and resumed from the saved file, behaviour and floors on 1K
#   fresh samples -> experiments/pcfg_anchors/results/a0.json; checkpoints and a0.log in results/large/a0/.
# a1 PRIOR: 50K iterations (~10 min plus compile at the a0 rate) with a 25-min deadline. Finished: results/a1_PRIOR.json;
#   log in results/large/a1/a1_PRIOR.log.
# a1c: controls on the three pretrained models (patch, B nulls, probes, bars), no deadline (est. 10 min, not measured)
#   -> results/a1c.json; log results/large/a1c/a1c.log.
# a2 ARM...: fine-tunes, each arm 10K iterations plus its readings; skips arms whose results/a2_ARM.json exists;
#   22-min deadline (leaves time for one arm's readings and the save). Log results/large/a2/a2.log.
# a1 and a2: on exit 3 (deadline: saved, not finished) the job resubmits itself with the same arguments from
#   $SLURM_SUBMIT_DIR, at most 3 times (count in E55_RESUBMITS); the next job resumes from the saved files.
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
usage="usage: sbatch cluster/pcfg_anchors.sbatch a0 | a1 {LOW,HIGH,NEVER} | a1c | a2 ARM [ARM ...]"
stage=${1:-}
args=("${@:2}")
case "$stage ${#args[@]}" in
  "a0 0" | "a1c 0") ;;
  "a1 1") case "${args[0]}" in LOW | HIGH | NEVER) ;; *) echo "$usage" >&2; exit 2 ;; esac ;;
  "a2 "*) [ "${#args[@]}" -ge 1 ] || { echo "$usage" >&2; exit 2; } ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
resubmits=${E55_RESUBMITS:-0}
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$* resubmits=$resubmits" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
out=experiments/pcfg_anchors/results/large/$stage
mkdir -p "$out"
run=(.venv/bin/python experiments/pcfg_anchors/run.py "$stage")
case "$stage" in
  a0) cmd=("${run[@]}"); logf=$out/a0.log ;;
  a1c) cmd=("${run[@]}"); logf=$out/a1c.log ;;
  a1) cmd=("${run[@]}" --prior "${args[0]}" --deadline-min 25); logf=$out/a1_${args[0]}.log ;;
  a2) cmd=("${run[@]}" --deadline-min 22 --arm "${args[@]}"); logf=$out/a2.log ;;
esac
rc=0
"${cmd[@]}" 2>&1 | tee -a "$logf" || rc=$?
if [ "$rc" = 3 ]; then
  if [ "$resubmits" -ge 3 ]; then
    echo "=== DEADLINE (exit 3) after $resubmits resubmits: not resubmitting; resume by hand ==="
    exit 3
  fi
  cd "$SLURM_SUBMIT_DIR"
  E55_RESUBMITS=$((resubmits + 1)) sbatch --job-name="$SLURM_JOB_NAME" "$0" "$stage" "${args[@]}"
  echo "=== DEADLINE (exit 3): resubmitted ($((resubmits + 1)) of 3) ==="
  exit 0
fi
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/ ==="
exit $rc
