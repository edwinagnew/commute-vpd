#!/bin/bash
# Cell P3.6 run 2 (qwen_cheap_readings): the top-set overlap of the 19 saved Qwen3-1.7B updates, each beside a random change
# at its own per-matrix norms; E92's columns for the 8 updates E92 has no row for; the contrasts. Card
# experiments/qwen_cheap_readings/card_run2.md; rules experiments/qwen_cheap_readings/results/predictions_run2.json
# (run2.py refuses unless it is committed unchanged; settings must equal run2.py's Config).
#   cd <clone root> && ~/commute-agents/fleet/link-data . && sbatch --comment='E### run2' cluster/qwen_cheap_readings_run2.sbatch
# Stage 0 (the do-nothing update reads overlap exactly 1.0; E92's columns reproduce on anchor 1 seed 0), stage 1 (the floor
# draws at F_s0 and a2_s42, the band), stage 2 (19 updates), stage 3 (CPU join and contrasts); each prints "STAGE n DONE"
# after results/run2_stage<n>.json. A filed stop prints "STOP: ..." and exits 1. Exit 3: stage 2 projected past 100 min
# and stopped between updates; resubmit the same command (stages 0-1 and the updates written are skipped).
# Inputs on NFS, read only: the 19 update files as run 1 (93 GB), E92's row inputs and stored reads, run 1's stage 1 and 2,
# Qwen3-1.7B from the HF cache, GSM8K test. Writes only experiments/qwen_cheap_readings/results/run2_stage{0,1,2,3}.json.
# Time on an L40, from run 1's log (job 59127: one overlap read of the 103 gold rows, random draw included, 19 s) and E92's
# (qwen_comparators/results/run.log, L40: read_state 86 s on gold, 111-125 s on TOFU, 50-74 s on anchor 1; a 5.64 GB fp32
# file ~24 s): stage 0 ~2.5 min; stage 1 8 draws ~4 min; stage 2 19 loads ~7 min, 19 real reads ~6 min, 68 draws ~22 min
# (gold 19 s, TOFU ~24 s, anchor 1 ~12 s, scaled by rows), 8 E92 reads ~12.5 min; ~55 min in all at 4 draws (~41 at 2).
# GPU: E92's peak 21.3 GB with a state on the card, plus one random W of 5.6 GB: ~27 GB of the L40's 46. Host: run 1's
# peak 10.5 GB, E92's 11.0 GB (updates load straight to the card): 24G. 16 CPUs as run 1: its 19 s includes the CPU draw.
#SBATCH --job-name=commute-qwen_cheap_readings_run2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:50:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/qwen_cheap_readings/results/predictions_run2.json || { echo "file the predictions first"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_cheap_readings/run2.py
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
