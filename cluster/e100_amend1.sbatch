#!/bin/bash
# E100 Amendment 1: the sum of single-part reverts on E58's four open-pair maps at seeds 3-7, read fed / written / between
# against the bars filed from seeds 0-2. No training. Plan: reports/plans/2026-11-05_plan_e100-amend1.md.
# Rules: experiments/fed_or_written/results/predictions_amend1.json (amend1.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/e100_amend1.sbatch
# Stages: 0 E62 copy seed 0's sum re-read against E100's 0.26277 within 1e-3 (exit 1 on a miss); 1 the 20 fresh states
# (sha256 against open_pair_instruments/results/seeds.json). Each prints "STAGE <n> DONE" when complete. A resubmit re-runs
# stage 0 and skips the fresh states already in amend1.json.
# Writes experiments/fed_or_written/results/amend1.json, table_amend1.json, amend1.out. Deletes nothing, commits nothing.
# Expected about 3 min on a debug L40, from E100's run.out (same card): start-up and file checks about 1.5 min; E100 read a
# key-table state's 85 reads in ~2.5 s (stage2.json `seconds`), this reads 29 per state, so ~1 s each plus one W0 base per
# site (4) and ~113 MB loaded and hashed per state; peak host RSS in E100 was 6.1 GB.
#SBATCH --job-name=commute-e100-amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/fed_or_written/amend1.py 2>&1 | tee -a experiments/fed_or_written/results/amend1.out || rc=$?
echo "=== FINISHED E100 Amendment 1 (exit $rc) — results in $COMMUTE_REPO/experiments/fed_or_written/results/ ==="
exit $rc
