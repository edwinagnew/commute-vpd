#!/bin/bash
# P1.58 run 2 (E###) the gain-only share on unseen cues and keys: one level per job (N or V), on one L40 (debug QoS).
# Card experiments/gain_share/card_unseen.md; rules in experiments/gain_share/results/unseen/predictions.json
# (unseen.py refuses unless it is committed unchanged). Submit both at once from the clone root:
#   sbatch --comment='E### gain share unseen cues: N' --time=00:50:00 cluster/gain_share_unseen.sbatch N
#   sbatch --comment='E### gain share unseen cues: V' cluster/gain_share_unseen.sbatch V
# Exit 1 at any miss (the log's STOP line says which, the stage's JSON holds "stopped"); a rerun skips every
# (state, level) already written. The V job's E62 stage needs N's gate (results/unseen/controls_N.json done): if N's
# controls are not done when V reaches it (expected only if N started over ~25 min after V), V exits with an
# AssertionError naming the file, and a resubmit of V reads only E62.
# Time, from run 1's measured s/step on an L40 (results/stage*/*.json) times 2000 steps per fit, 11 fits per level:
#   N: 2 x 160 s (x2, x10) + 9 x 122 s + reads and set-up ~3 min = about 26 min (limit 50 min)
#   V: 2 x 322 s + 9 x 282 s + reads and set-up ~4 min = about 58 min (limit 1 h 45 min, under the 2 h cap)
# Peak host memory 6.1 GB and GPU 10.6 GB in run 1. Writes only under experiments/gain_share/results/.
#SBATCH --job-name=commute-p1-58-gain-unseen
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
[[ "${1:-}" == N || "${1:-}" == V ]] || { echo "usage: sbatch ... cluster/gain_share_unseen.sbatch N|V" >&2; exit 2; }
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
large=experiments/gain_share/results/large
[ -e "$large" ] || "$HOME/commute-agents/fleet/link-data" "$COMMUTE_REPO"
mkdir -p experiments/gain_share/results/unseen
log=experiments/gain_share/results/unseen/run_$1.log
rc=0
.venv/bin/python experiments/gain_share/unseen.py "$1" 2>&1 | tee -a "$log" || rc=${PIPESTATUS[0]}
echo "=== FINISHED P1.58 run 2 (unseen cues) $1 (exit $rc) — results in $COMMUTE_REPO/experiments/gain_share/results/unseen/ ==="
exit $rc
