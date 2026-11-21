#!/bin/bash
# E_TBD2 (cell P2.54) run 2: the low-prior copy (selection, 3 fine-tunes, matched random updates), the head switch-off
# on the new states, and every proxy on the four corners. Needs run 1 (cluster/prior_mech_2x2.sbatch) finished: submit
# with --dependency=afterok:<run 1's job id>. Plan reports/plans/2026-11-21_plan_prior-mech-2x2.md; rules
# experiments/prior_mech_2x2/results/predictions_stage2.json (train.py refuses unless it is committed unchanged; PRED_SHA
# below = its sha256: recompute with `git show HEAD:<that path> | sha256sum` after setting its "experiment").
#   sbatch --comment=E_TBD2 --dependency=afterok:<run1> cluster/prior_mech_2x2_stage2.sbatch
# Time, from measured items (estimate, not yet run): model load ~40 s and split bases 10 s (E237 job 62504 log);
# E147's fine-tune ran 300 steps in ~9 s on an L40 (job 57955, 19:27:21 step 30 -> 19:27:30 step 300), x3; E147's
# stage 1 read every reader (E136, E98, holes_fill) on one key state in ~17 s (job 57961, 24 states in 12.8 min), so
# holes_fill alone <= 17 s x 21 items; r <= 11 s x 6 new items (E237 stage1 read_seconds); head switch-off ~5 s x 3;
# the selection is one forward over 20000 41-token windows (seconds). Sum ~10 min; debug's 30 min cap holds with 3x
# margin. $0. Peak GPU ~10 GB (E221/E226 with the split; holes_fill's steering grid is smaller).
#SBATCH --job-name=commute-prior-mech-2
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
for d in refit_96key induction_cue open_pair transport_test prior_mech_2x2; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
rc=0
( PRED_SHA=9b44981f9828b1b06036daaeddda9fc5a65e68d2fc4e18fdd7eb00df34ab6e1a .venv/bin/python experiments/prior_mech_2x2/train.py ) 2>&1 \
    | tee -a experiments/prior_mech_2x2/results/run2.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E_TBD2 run 2 (exit $rc) — results in $COMMUTE_REPO/experiments/prior_mech_2x2/results/stage2/ ==="
exit $rc
