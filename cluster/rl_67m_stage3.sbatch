#!/bin/bash
# E222 (cell P2.41 stage 3): the re-gating share and the scorecard baselines on E220's 27 saved states. Plan:
# reports/plans/2026-11-19_plan_rl-67m.md (Stage 3 (E222)). Settings, expected values and rules:
# experiments/rl_67m/results/predictions_E222.json (run_stage3.py refuses unless it is committed unchanged; it also
# checks E220's predictions_E220.json, so run on the cluster from a clone, not on a pod with one PRED_SHA).
# One job per argument; submit from the clone root:
#   sbatch --comment=E222 --time=00:30:00 cluster/rl_67m_stage3.sbatch ctl                          # first and alone
#   sbatch --comment=E222 --time=00:30:00 --dependency=afterok:<ctl id> cluster/rl_67m_stage3.sbatch 1   # and 2
#   sbatch --comment=E222 --time=00:20:00 --dependency=afterok:<ctl id> cluster/rl_67m_stage3.sbatch anchor
# Times on an L40, each a measured item x its count: model load ~2 min (E218); E211's ctl 4 min (log 61444); per
# behaviour reader ~2.5 min (rows, demo pool, reader build: E211's read+zero ~1 min, log 61448, plus the anchor's
# Wikipedia stream ~1 min, E220); per state at most 59 s (E211's T2 states, log 61448, 500 rows of 127 tokens; these are
# 200 of 65). ctl ~14 min (2 + 4 + 3 x (2.5 + 1)); jobs 1 and 2 ~17 min (2 + 2.5 + 13 states x 59 s); anchor ~8 min
# (2 + 2.5 + 4 x 59 s). Host memory as E211's jobs (peak 6.1 GB).
# Inputs: the model in artifacts/vpd/; artifacts/sets/{val_0_64,val_0_8192,train_0_32768}.npy;
# experiments/rl_67m/results/{stage0,stage1,predictions,predictions_E220,stage2b_*}.json, results/large/stage1_samples.pt,
# results/large/stage2b/*.pt (E220's 27 states); experiments/transport_test/results/large/windows.pt; French Wikipedia
# streamed (the anchor's demonstration pool); ctl also E211's inputs (E53's two states, E136/E143/E98/update_svd records).
# Writes experiments/rl_67m/results/stage3_<job>.json and results/stage3/<behaviour>_<arm>_s<seed>.json (one per state).
#SBATCH --job-name=commute-rl67m-s3
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as E211: l40-worker's /ephemeral was full (2026-11-18)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/rl_67m/results/predictions_E222.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/rl_67m/run_stage3.py "$1"
