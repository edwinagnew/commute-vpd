#!/bin/bash
# E_TBD (cell P2.54) run 1: copy head (2,4) switched off on E237's 15 key states and E62's six, nothing trained.
# Plan reports/plans/2026-11-21_plan_prior-mech-2x2.md; rules experiments/prior_mech_2x2/results/predictions.json
# (run.py refuses unless it is committed unchanged; PRED_SHA below = its sha256). Submit from the clone root:
#   sbatch --comment=E_TBD cluster/prior_mech_2x2.sbatch
# Time: model load ~40 s (E237 job 62504: start 17:05:09, split bases logged 17:05:54, a heavier load than this one);
# per state 7 reads of the held-out rows (F, F with (2,4) off, five random heads off) against E237's 10-11 s for one
# share read of 4+ masked forwards on the same rows (stage1.json read_seconds), so <= ~15 s a state x 21 states
# + M's reads on 7 tasks: ~6 min, well inside debug's 30 min. Peak GPU < 4 GB (one 67M model, batch 64, no Gamma).
# Order inside the job: control (exit 1 on a miss), E62 repeat (exit 1 on a miss), key states, summary.
#SBATCH --job-name=commute-prior-mech
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
for d in refit_96key induction_cue open_pair; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
rc=0
( PRED_SHA=86d44698e205c5998047d1a616496c4982157e17ee4d77d7f0d02cdac308feeb .venv/bin/python experiments/prior_mech_2x2/run.py ) 2>&1 \
    | tee -a experiments/prior_mech_2x2/results/run1.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E_TBD run 1 (exit $rc) — results in $COMMUTE_REPO/experiments/prior_mech_2x2/results/stage1/ ==="
exit $rc
