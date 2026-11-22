#!/bin/bash
# E_TBD_A (cell P2.57) run A: key copy against the prior-matched cipher at 10 seeds (E147's recipe), every cheap
# reading, the seed-level rank test. Needs P2.54's run 1 (E249) and E250 finished (their readers and controls are used).
# Rules experiments/prior_matched/results/predictions_a.json (run_a.py refuses unless it is committed unchanged; set
# PRED_SHA below after setting its "experiment": `git show HEAD:experiments/prior_matched/results/predictions_a.json | sha256sum`).
#   sbatch --comment=E### cluster/prior_matched_a.sbatch
# Time, from E250's measured log (experiments/prior_mech_2x2/results/run2.log): load and repeats 26 s; khead's M control
# 83 induction reads ~170 s; 20 fine-tunes x 11 s = 220 s; split bases 13 s; 27 items x ~42 s (holes_fill and r ~30 s,
# khead's 85 head sets ~12 s) = 1134 s; khead's M rows ~25 s. Sum ~1590 s, about 27 min; the 60 min limit leaves 2x. $0.
# Peak GPU ~8 GB (E250 7.74 GB).
#SBATCH --job-name=commute-prior-matched-a
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # caches in /ephemeral/$USER; E12's Pile rows are artifacts/sets/train_0_32768.npy
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for d in refit_96key induction_cue open_pair transport_test prior_mech_2x2 prior_matched; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
rc=0
( PRED_SHA=SET_ME .venv/bin/python experiments/prior_matched/run_a.py ) 2>&1 \
    | tee -a experiments/prior_matched/results/run_a.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED prior-matched run A (exit $rc) — results in $COMMUTE_REPO/experiments/prior_matched/results/a/ ==="
exit $rc
