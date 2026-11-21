#!/bin/bash
# E_TBD (cell P2.54) run 2: the cued copy, a low-prior copy built as its own task (bands, selection, 3 fine-tunes,
# matched random updates, head labels, every proxy, the 2x2). Needs P2.54's run 1 (E249) and E250 finished (their
# results are read; khead's run 3 is optional: its k-head values for the key corners read nan until it has run).
# Plan reports/plans/2026-11-21_plan_cued-copy.md; rules experiments/cued_copy/results/predictions.json (run.py refuses
# unless it is committed unchanged; set PRED_SHA below to its sha256 after setting its "experiment":
# `git show HEAD:experiments/cued_copy/results/predictions.json | sha256sum`).
#   sbatch --comment=E### cluster/cued_copy.sbatch
# Time, from E250's measured log (experiments/prior_mech_2x2/results/run2.log): model load and both repeats 26 s
# (13:58:39 -> 13:59:05); the selection forward over 20,000 windows 7 s, so 40,000 a band ~14 s, 3 bands at most ~42 s;
# a 300-step fine-tune 11 s (13:59:12 -> :24 -> :35 -> :46), x3 = 33 s; split bases 13 s; one new item's holes_fill and r
# reads ~30 s (14:04:16 -> 14:04:45 -> 14:05:16), x6 = 180 s; run 1's head read of 7 head sets took 1 s per item, so
# khead's 84 sets ~12 s per item, x6 plus M's rows once ~84 s; run 1's control ~20 s. Sum ~400 s, about 7 min (E250:
# 7 min wall for 3 fine-tunes and 21 items). debug's 30 min cap holds with 4x margin. $0. Peak GPU ~8 GB (E250 7.74 GB).
#SBATCH --job-name=commute-cued-copy
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # caches in /ephemeral/$USER; E12's Pile rows are artifacts/sets/train_0_32768.npy
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for d in refit_96key induction_cue open_pair transport_test prior_mech_2x2 cued_copy; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
rc=0
( PRED_SHA=SET_ME .venv/bin/python experiments/cued_copy/run.py ) 2>&1 \
    | tee -a experiments/cued_copy/results/run.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED cued copy (exit $rc) — results in $COMMUTE_REPO/experiments/cued_copy/results/ ==="
exit $rc
