#!/bin/bash
# E135 (cell P2.5): does steering separate copy from lookup as widely as the share, once it has per-key vectors and a
# tuned strength? Plan: reports/plans/2026-11-09_plan_e135-steering-copy-lookup.md. Rules:
# experiments/steering_copy_lookup/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && sbatch cluster/steering_copy_lookup.sbatch
# Stage 0 (the control, alone) exits 1 on a miss. Prints "STAGE n DONE" after each results file.
# Inputs on NFS (gitignored): experiments/induction_cue/results/large/{windows.pt,copy_s*.pt,lookup_s*.pt},
# experiments/transport_test/results/large/alive.pt, the target model in artifacts/vpd/. Writes
# experiments/steering_copy_lookup/results/{stage0,stage1}.json only; commits nothing.
# Expected on an L40: about 7 min (model load ~2 min, stage 0's six route splits on 200 windows ~1 min, stage 1's six
# route splits on 100 windows plus the steering forwards ~3 min; E62's route split took ~9 s per state on an L40,
# results/large/run.out lines 351-356). The limit leaves room for a slow model load.
#SBATCH --job-name=commute-steering-copy-lookup
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --comment=E135
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/steering_copy_lookup/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/steering_copy_lookup/run.py
