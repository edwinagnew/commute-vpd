#!/bin/bash
# Cell P2.68 (E###): whole-group head CMAP after Prakash et al. 2024 on the 67M. Spec
# reports/plans/2026-11-23_plan_p2-68-cmap-spec.md; rules experiments/cmap/results/predictions.json (run.py refuses unless
# it is committed unchanged). Arguments go to run.py; the authors' read-only checkout is passed here.
# Needs the venv's `cmap` dependency group (baukit, peft): `uv sync --locked --inexact --no-default-groups --group dev
# --group gpu --group cmap` (--inexact so other groups' packages stay). Submit from the clone root, after
# ~/commute-agents/fleet/link-data . (results/large symlinks):
#   sbatch --comment=E### cluster/cmap.sbatch --stage 0              (alone and first; exit 1 on a missed control)
#   sbatch --comment=E### cluster/cmap.sbatch --stage 1 --shard 0 4  (and 1 4, 2 4, 3 4, after stage 0 passes)
# E305 (cell P2.80, the five seed-2 states; stage 0's controls then the items, in one job):
#   sbatch --comment=E305 --export=ALL,CMAP_ENTRY=p2_80.py cluster/cmap.sbatch
#SBATCH --job-name=commute-cmap
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python -c 'import baukit, peft' || { echo "the cmap dependency group is not installed (see the header)"; exit 1; }
for d in refit_96key share_untrained induction_cue known_3seeds; do
  test -L experiments/$d/results/large || { echo "experiments/$d/results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
done
.venv/bin/python "experiments/cmap/${CMAP_ENTRY:-run.py}" --authors "${CMAP_AUTHORS:-$HOME/commute-data/scratch/regating-lead/p268-spec/authors-code}" "$@"
