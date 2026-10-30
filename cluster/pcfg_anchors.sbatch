#!/bin/bash
# E55 Stage A: PCFG minGPT anchors. Plan: reports/plans/2026-10-29_plan_reuse-stage1.md (Stage A).
#   cd ~/commute-agents/<role>
#   sbatch cluster/pcfg_anchors.sbatch a0                                   # rehearsal, ~5-10 min
#   sbatch --job-name=commute-e55-a1-LOW cluster/pcfg_anchors.sbatch a1 LOW  # one job per prior: LOW, HIGH, NEVER
#   sbatch --job-name=commute-e55-a1c-patch cluster/pcfg_anchors.sbatch a1c-patch  # after all three a1 jobs
#   sbatch --job-name=commute-e55-a1c-probe cluster/pcfg_anchors.sbatch a1c-probe  # independent of a1c-patch
#   (file a2_bars in predictions.json, dated, from results/a1c_patch.json; a2 exits 2 until then)
#   sbatch --job-name=commute-e55-a2-1 cluster/pcfg_anchors.sbatch a2 N1 N2 N3 W2 R   # arms in order, ~3 min each
#   (two a2 jobs must never list the same arm: both would write its _train.pt)
#   sbatch --job-name=commute-e55-amend1 cluster/pcfg_anchors.sbatch amend1      # Amendment 1, ~15-20 min
#   sbatch --job-name=commute-e55-amend2 --time=00:10:00 cluster/pcfg_anchors.sbatch amend2  # Amendment 2, ~3-5 min
#   sbatch --job-name=commute-e55-amend3 --time=00:10:00 cluster/pcfg_anchors.sbatch amend3  # Amendment 3, ~6-8 min
#   sbatch --job-name=commute-e55-amend4 --time=00:10:00 cluster/pcfg_anchors.sbatch amend4  # Amendment 4, ~2-5 min (cost not measured)
#   sbatch --job-name=commute-e55-amend5c --time=00:10:00 cluster/pcfg_anchors.sbatch amend5 controls  # Amendment 5 controls, ~2-3 min (estimated)
#   sbatch --job-name=commute-e55-amend5 cluster/pcfg_anchors.sbatch amend5  # (after amend5c finishes: both write amend5.json) Amendment 5 arms, ~12-16 min (estimated)
#   sbatch --job-name=commute-e55-amend5b cluster/pcfg_anchors.sbatch amend5b  # Amendment 5b (WNf seeds 1-3), ~4-6 min (estimated); before c1
#   sbatch --job-name=commute-e55-c1c cluster/pcfg_anchors.sbatch c1 controls  # Stage C1 controls, <= 7 min (estimated)
#   sbatch --job-name=commute-e55-c1 cluster/pcfg_anchors.sbatch c1  # (after c1c finishes and its controls are read: both write c1.json) Stage C1 states, <= 119 min = up to 6 jobs of 22 min (estimated; 7 allowed with resubmits, tight)
# a0: per prior, 500 pretraining iterations stopped at 250 and resumed from the saved file, behaviour and floors on 1K
#   fresh samples -> experiments/pcfg_anchors/results/a0.json; checkpoints and a0.log in results/large/a0/.
# a1 PRIOR: 50K iterations (~10 min plus compile at the a0 rate) with a 25-min deadline. Finished: results/a1_PRIOR.json;
#   log in results/large/a1/a1_PRIOR.log.
# a1c-patch: patch references, B nulls, l* and the A2 bars on the three pretrained models -> results/a1c_patch.json.
# a1c-probe: probe margins, m_plus, possession and PA4 -> results/a1c_probe.json. Each has no deadline (cost not
#   measured; the log gives each section's elapsed time), skips when its file exists; log results/large/<stage>/.
# a2 ARM...: fine-tunes, each arm 10K iterations plus its readings; skips arms whose results/a2_ARM.json exists;
#   22-min deadline (leaves time for one arm's readings and the save). Log results/large/a2/a2.log.
# amend1: experiments/pcfg_anchors/amend1.py (predictions_amend1.json): controls (formula, synthetic answerers, N1 10K,
#   LOW), the saved W states,
#   then the W1 and W1s1 trajectory re-runs -> results/amend1.json; 22-min deadline. Log results/large/amend1/amend1.log.
#   A missed control writes results/amend1_control_miss.json (not amend1.json) and exits 1; a rerun re-reads the controls.
# amend2: experiments/pcfg_anchors/amend2.py (predictions_amend2.json): the third-rule test on 20,000 fresh rows and
#   ~9K rows in unseen (a, b) cells, no training: model-free controls, then N1 10K and LOW on both sets, then the
#   saved W states and amend1's trajectory states on both -> results/amend2.json; a control miss -> results/amend2_control_miss.json,
#   exit 1. No deadline (one pass, ~3-5 min). Log results/large/amend2/amend2.log.
# amend3: experiments/pcfg_anchors/amend3.py (predictions_amend3.json): natural rows by (a, b) pair (LOW, N1 10K, W states)
#   and swap size on Stage A's C_Te = 1 rows (bases and every A2 state), no training: model-free controls, then LOW and
#   N1 controls -> results/amend3.json; a control miss -> results/amend3_control_miss.json, exit 1. No deadline (one
#   pass; ~5 min of it is CPU sampling of rare pairs). Log results/large/amend3/amend3.log.
# amend4: experiments/pcfg_anchors/amend4.py (predictions_amend4.json; refuses a draft): patches on swap-built rows
#   (Stage A's C_Te = 1 rows, natural donors) and on off-label swap-built rows, no training: model-free controls, then
#   LOW and every state's natural P (reproduces a2) -> results/amend4.json; a control miss ->
#   results/amend4_control_miss.json, exit 1. No deadline (one pass). Log results/large/amend4/amend4.log.
# amend5 [controls]: experiments/pcfg_anchors/amend5.py (predictions_amend5.json; refuses a draft): model-free controls,
#   LOW W0 controls and reference; with "controls" it stops there (~1 min CPU build + model reads). Without, then 7
#   fine-tunes from LOW (ID s0, WN s0-s3, WNf, ID s1; 2K steps each at ~12 ms/step = ~25 s plus compile, data and
#   three states read's ~1 min: estimated ~1.5-2 min per arm) -> results/amend5.json after each arm; states in
#   results/large/amend5/; 22-min deadline, exit 3 resubmits and resumes. A control miss ->
#   results/amend5_control_miss.json, exit 1. Log results/large/amend5/amend5.log.
# amend5b: experiments/pcfg_anchors/amend5b.py (predictions_amend5b.json; refuses a draft): amend5's model-free and W0
#   controls, then WNf seeds 1-3 (lr 1e-4, 2K steps, saved at 500, 1K, 2K in results/large/amend5b/) -> results/amend5b.json
#   after each arm (amend5.json untouched); 22-min deadline, exit 3 resubmits. A control miss -> results/amend5b_control_miss.json,
#   exit 1. Log results/large/amend5b/amend5b.log.
# c1 [controls]: experiments/pcfg_anchors/c1.py (predictions_c1.json; refuses a draft): model-free controls, then with
#   "controls" the instruments' controls alone (pruning on LOW, sign flipped; scratch = NEVER reverse fine-tuned at 1e-5
#   and 1e-6, <= 10K steps each at ~12.5 ms/step: <= 7 min). Without, it refuses (exit 2) until c1.json holds the controls,
#   then per listed state (36 with reverse fine-tuning, <= 3.3 min each at the cap; 2 steering only, ~10 s): pruning,
#   steering, reverse fine-tuning -> results/c1.json after each state. 22-min deadline, exit 3 resubmits (up to 6 times
#   for c1) and resumes; a reverse fine-tune cut by the deadline is rerun. A model-free miss -> results/c1_control_miss.json,
#   exit 1. Log results/large/c1/c1.log.
# a1, a2, amend1, amend5, amend5b and c1: on exit 3 (deadline: saved, not finished) the job resubmits itself with the same arguments from
#   $SLURM_SUBMIT_DIR, at most 3 times (c1: 6) (count in E55_RESUBMITS); the next job resumes from the saved files.
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
usage="usage: sbatch cluster/pcfg_anchors.sbatch a0 | a1 {LOW,HIGH,NEVER} | a1c-patch | a1c-probe | a2 ARM [ARM ...] | amend1 | amend2 | amend3 | amend4 | amend5 [controls] | amend5b | c1 [controls]"
stage=${1:-}
args=("${@:2}")
case "$stage ${#args[@]}" in
  "a0 0" | "a1c-patch 0" | "a1c-probe 0" | "amend1 0" | "amend2 0" | "amend3 0" | "amend4 0" | "amend5 0" | "amend5b 0" | "c1 0") ;;
  "amend5 1" | "c1 1") [ "${args[0]}" = controls ] || { echo "$usage" >&2; exit 2; } ;;
  "a1 1") case "${args[0]}" in LOW | HIGH | NEVER) ;; *) echo "$usage" >&2; exit 2 ;; esac ;;
  "a2 "*) [ "${#args[@]}" -ge 1 ] || { echo "$usage" >&2; exit 2; } ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
resubmits=${E55_RESUBMITS:-0}
max_resubmits=3
[ "$stage" = c1 ] && max_resubmits=6
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$* resubmits=$resubmits" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
out=experiments/pcfg_anchors/results/large/$stage
mkdir -p "$out"
run=(.venv/bin/python experiments/pcfg_anchors/run.py "$stage")
case "$stage" in
  a0) cmd=("${run[@]}"); logf=$out/a0.log ;;
  a1c-patch | a1c-probe) cmd=("${run[@]}"); logf=$out/$stage.log ;;
  a1) cmd=("${run[@]}" --prior "${args[0]}" --deadline-min 25); logf=$out/a1_${args[0]}.log ;;
  a2) cmd=("${run[@]}" --deadline-min 22 --arm "${args[@]}"); logf=$out/a2.log ;;
  amend1) cmd=(.venv/bin/python experiments/pcfg_anchors/amend1.py --deadline-min 22); logf=$out/amend1.log ;;
  amend2) cmd=(.venv/bin/python experiments/pcfg_anchors/amend2.py); logf=$out/amend2.log ;;
  amend3) cmd=(.venv/bin/python experiments/pcfg_anchors/amend3.py); logf=$out/amend3.log ;;
  amend4) cmd=(.venv/bin/python experiments/pcfg_anchors/amend4.py); logf=$out/amend4.log ;;
  amend5) cmd=(.venv/bin/python experiments/pcfg_anchors/amend5.py --deadline-min 22 ${args[@]:+--controls-only}); logf=$out/amend5.log ;;
  amend5b) cmd=(.venv/bin/python experiments/pcfg_anchors/amend5b.py --deadline-min 22); logf=$out/amend5b.log ;;
  c1) cmd=(.venv/bin/python experiments/pcfg_anchors/c1.py --deadline-min 22 ${args[@]:+--controls-only}); logf=$out/c1.log ;;
esac
rc=0
"${cmd[@]}" 2>&1 | tee -a "$logf" || rc=$?
if [ "$rc" = 3 ]; then
  if [ "$resubmits" -ge "$max_resubmits" ]; then
    echo "=== DEADLINE (exit 3) after $resubmits resubmits: not resubmitting; resume by hand ==="
    exit 3
  fi
  cd "$SLURM_SUBMIT_DIR"
  E55_RESUBMITS=$((resubmits + 1)) sbatch --job-name="$SLURM_JOB_NAME" "$0" "$stage" "${args[@]}"
  echo "=== DEADLINE (exit 3): resubmitted ($((resubmits + 1)) of $max_resubmits) ==="
  exit 0
fi
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/pcfg_anchors/results/ ==="
exit $rc
