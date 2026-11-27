#!/bin/bash
# E314 (cell P2.85): two cheap SAE proxies on the 67M at E234's base-only SAEs (error-term share at the 8 block-output
# sites; Kissane et al.'s explained-variance gap at all 13), on the 48 known3.json states and one matched random update
# each. Plan reports/plans/2026-11-26_plan_e314-sae-proxies.md; rule experiments/sae_proxies/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/sae-lead && sbatch --comment=E314 cluster/sae_proxies.sbatch
# Stage 0 (zero update, run-to-run floor, full patch, E234's base FVE, the 48 gains) prints "STAGE 0 DONE"; stage 1
# writes each state to results/results.json as it lands and prints "STAGE 1 DONE"; a stop prints "STOP: ..." and exits 1.
# A rerun resumes from results/results.json.
# Inputs (NFS, read only): the model in artifacts/vpd/, E234's 13 SAEs (sae_train/results/large/scored, ~100 MB each),
# Pile val rows (artifacts/sets/val_0_8192.npy), E53's cached Pile rows, the 48 update files (57-113 MB each, ~3.5 GB)
# and the windows named in share_untrained and refit_96key inventories.
# Expected on a debug L40, estimated (not measured for this script): model, SAEs and the 16 row sets about 3 min
# (E136/E147 load 3 min); E234's FVE read at 2 x 262k Pile tokens about 1 min; 48 gains about 2 min (one file load,
# ~1 s on NFS, plus a 100-300-row forward each); stage 1, 96 updates at ~4 s (file load, 3 task-row forwards, one
# 128-row Pile forward, 13-site SAE passes on <12k tokens) about 7 min. Total about 15 min; limit 60 min.
# Memory: activations of 13 sites x 2 models x <12k tokens are under 2 GB of GPU; SAEs 1.3 GB; host 24G is ample.
#SBATCH --job-name=commute-sae-proxies
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --comment=E314
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/sae_proxies/results/predictions.json || { echo "file the predictions first"; exit 1; }
rc=0
( .venv/bin/python experiments/sae_proxies/run.py ) 2>&1 | tee -a experiments/sae_proxies/results/run.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E314 (exit $rc) — results in $COMMUTE_REPO/experiments/sae_proxies/results/ ==="
exit $rc
