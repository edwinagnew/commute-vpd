#!/bin/bash
# E309 (cell P2.79): the reading refactor's new-code reproduction, writer 2b's readings, gpu part
# (experiments/reading_refactor/new_b.py): steering E135 / E143 / the panel's, ADL column D, the fresh E_b passes and the
# copy24_s0 neuron K scan through the new functions, to experiments/reading_refactor/results/new_b.json, keyed as old.py
# keys them; compare.py then holds each against old_A / old_B. Run it on the card type old.sbatch ran on (L40).
# Nothing is trained; nothing large is written.
#   Submit from the root of the checkout at the p2-79-refactor commit that holds new_b.py:
#   sbatch --comment='E309' experiments/reading_refactor/new_b.sbatch
# Inputs on NFS, read only (fleet/link-data): artifacts/vpd, artifacts/sets, saved updates and windows under
# experiments/*/results/large. Prints "NEW_B gpu DONE; errors: ..." at the end; a group that raised is written as
# _error/<group> and the job exits 1.
# Time: not measured. The same readings in old.sbatch's groups steering_e135 + steering_e143 + panel + eb_fresh; use
# old_A.json's `_seconds` for those four groups plus 25% as the limit once run A has run.
#SBATCH --job-name=commute-e309-new-b
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the checkout's root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS commit=$(git rev-parse HEAD)" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/reading_refactor/new_b.py --part gpu
