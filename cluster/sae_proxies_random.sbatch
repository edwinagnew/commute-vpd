#!/bin/bash
# E318 (cell P2.88): Fig 3's random updates (E302's draws, rebuilt from seed and signed_c) read through E314's error
# share and E317's reconstruction share. experiments/sae_proxies/random_ref.py; rule
# experiments/sae_proxies/results/predictions_random.json (refused unless committed unchanged).
#   cd ~/commute-agents/sae-lead && sbatch --comment=E318 cluster/sae_proxies_random.sbatch
# Stage 0 (E314 error share read again; rebuilt size) prints "STAGE 0 DONE"; stage 1 writes each of the 144 draws to
# results/results_random.json and prints "STAGE 1 DONE"; a stop prints "STOP: ..." and exits 1; a rerun resumes.
# Expected: E317 (4 row forwards per update, 96 updates) ran in about 6 min, so 144 draws about 9-12 min; limit 60 min.
#SBATCH --job-name=commute-sae-proxies-random
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --comment=E318
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/sae_proxies/results/predictions_random.json || { echo "file the predictions first"; exit 1; }
rc=0
( .venv/bin/python experiments/sae_proxies/random_ref.py ) 2>&1 | tee -a experiments/sae_proxies/results/run_random.log || rc=${PIPESTATUS[0]}
echo "=== FINISHED E318 (exit $rc) — results in $COMMUTE_REPO/experiments/sae_proxies/results/ ==="
exit $rc
