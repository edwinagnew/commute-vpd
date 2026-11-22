#!/bin/bash
# E252 (cell P2.55) run 1: the k-head label. M's top k copy heads by E62's prefix score (k = 1, 2, 3, 5) switched
# off together, (2,4) with (1,1) as one more arm, and a k-matched null of 20 draws per k from outside the top 5 and
# (1,1), on all 27 saved states of runs 1-2 (E62's 6, E237's 15, E250's 3 low-copy states and 3 random updates).
# Nothing is trained. Rules experiments/prior_mech_2x2/results/predictions_khead.json (khead.py refuses unless it is
# committed unchanged; PRED_SHA below = its sha256: recompute with `git show HEAD:<that path> | sha256sum` after
# setting its "experiment"). Needs runs 1 and 2 finished (results/stage1/, results/stage2/head/, stage2/b.json).
#   sbatch --comment=E252 cluster/prior_mech_2x2_khead.sbatch
# Time (estimate, not yet run): E249's job took 34 s from start to DONE for 7 head sets on each of 21 states and M's 7
# tasks, 196 read sets, so <= 0.17 s a read set (model load included). Here 84 head sets (nothing off, the pair, 4 top-k
# sets, 18 + 20 + 20 + 20 null draws) on 27 items and M's 8 tasks = 2,940 read sets, plus ~120 in the controls (k = 1
# on 24 states, the floor at two batch sizes) and 84 cheap induction reads on M: ~3,150 x 0.17 s = ~9 min. Debug's
# 30 min cap holds with 3x margin. $0. Peak GPU < 4 GB (E249 3.55 GB, the same reader; one update in memory at a time).
# Order inside the job: control on M (exit 1 on a miss), the k = 1 repeat against the saved (2,4) shares within 2 x the
# measured floor (exit 1 on a miss), the 27 items, the summary.
#SBATCH --job-name=commute-prior-mech-k
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
for d in refit_96key induction_cue open_pair transport_test prior_mech_2x2; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
rc=0
( PRED_SHA=e322176bdb204be57b97cffe1de31b9de45199b49070577821d7a792a742978b .venv/bin/python experiments/prior_mech_2x2/khead.py ) 2>&1 \
    | tee -a experiments/prior_mech_2x2/results/run3.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E252 run 3 (exit $rc) — results in $COMMUTE_REPO/experiments/prior_mech_2x2/results/khead/ ==="
exit $rc
