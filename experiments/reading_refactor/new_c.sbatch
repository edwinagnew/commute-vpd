#!/bin/bash
# E309 (cell P2.79) stage "scans": the NEW code's side of the readings no earlier reproduction ran
# (experiments/reading_refactor/new_c.py): needed.scan (VPD) and needed_sae.scan on scan_sites, a fresh SAE write_sums
# pass with eb(leftovers=True/False), copy24_s0's neuron eb(leftovers=False), E135's per-key steering Rule. Writes
# experiments/reading_refactor/results/new_c.json; compare.py holds it against old_c_A / old_c_B. Same card type as
# old_c.sbatch (L40). Nothing is trained; nothing large is written.
#   Submit from the root of a checkout of p2-79-fix at or after this file's commit:
#   sbatch --comment='E309 stage: scans' experiments/reading_refactor/new_c.sbatch
# Inputs on NFS, read only (fleet/link-data): artifacts/vpd, the E234 SAEs, saved updates, windows and K arrays under
# experiments/*/results/large. Prints "NEW_C DONE; errors: ..."; a group that raised is written as _error/<group> and
# the job exits 1. Time: not measured; old run A's eb_fresh (one full neuron scan, two worlds) took 234 s, these scans
# are 2 pairs each plus E268's identity control: expected under 20 min, limit 1 h.
#SBATCH --job-name=commute-e309-new-c
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the checkout's root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS commit=$(git rev-parse HEAD)" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/reading_refactor/new_c.py
