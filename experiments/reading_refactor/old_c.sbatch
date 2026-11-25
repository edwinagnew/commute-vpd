#!/bin/bash
# E309 (cell P2.79) stage "scans": the OLD code's side (experiments/reading_refactor/old_c.py), run twice (tags A, B)
# for the old code's own floor; writes experiments/reading_refactor/results/old_c_<TAG>.json in the checkout it runs in.
# It must run against the OLD src: copy old_c.py and this file into the e309-old worktree (main at 62e4ca8d, which
# holds old.py) and submit from its root:
#   sbatch --comment='E309 stage: scans' experiments/reading_refactor/old_c.sbatch A
#   sbatch --comment='E309 stage: scans' experiments/reading_refactor/old_c.sbatch B
# then copy results/old_c_A.json and old_c_B.json back to p2-79-fix. Same card type (L40) as new_c.sbatch.
# Inputs on NFS, read only (fleet/link-data). Prints "OLD_C <TAG> DONE; errors: ..."; exits 1 if a group raised.
# Time: not measured; expected under 20 min (see new_c.sbatch), limit 1 h.
#SBATCH --job-name=commute-e309-old-c
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
TAG=${1:?usage: sbatch old_c.sbatch TAG}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the checkout's root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS tag=$TAG commit=$(git rev-parse HEAD)" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/reading_refactor/old_c.py --tag "$TAG"
