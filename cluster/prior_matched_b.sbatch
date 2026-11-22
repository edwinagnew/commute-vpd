#!/bin/bash
# E_TBD_B (cell P2.57) run B: a second prior-matched cipher on copy's keys (Amendment 6's rule, a new draw seed), 3
# seeds and matched random updates, read as run A, against run A's 10 copy seeds. Needs run A finished (its summary and
# copy items are read; run_b.py exits 1 at start otherwise): submit with --dependency=afterok:<run A's job id>.
# Rules experiments/prior_matched/results/predictions_b.json (run_b.py refuses unless it is committed unchanged; set
# PRED_SHA below after setting its "experiment": `git show HEAD:experiments/prior_matched/results/predictions_b.json | sha256sum`).
#   sbatch --comment=E### --dependency=afterok:<A> cluster/prior_matched_b.sbatch
# Time, from E250's measured log: load and repeats 26 s; khead's M control ~170 s; the build (key distributions, token
# counts, up to 4 draws and priors) ~50 s; 3 fine-tunes x 11 s = 33 s; split bases 13 s; 7 items x ~42 s = 294 s; khead's
# M rows ~12 s. Sum ~600 s, about 10 min; the 30 min limit leaves 3x. $0. Peak GPU ~8 GB.
#SBATCH --job-name=commute-prior-matched-b
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
for d in refit_96key induction_cue open_pair transport_test prior_mech_2x2 prior_matched; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
rc=0
( PRED_SHA=SET_ME .venv/bin/python experiments/prior_matched/run_b.py ) 2>&1 \
    | tee -a experiments/prior_matched/results/run_b.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED prior-matched run B (exit $rc) — results in $COMMUTE_REPO/experiments/prior_matched/results/b/ ==="
exit $rc
