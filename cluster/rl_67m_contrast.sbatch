#!/bin/bash
# E223 (cell P2.41 run 5): E222's re-gating share and scorecard baselines re-read on E220's 24 pick states with a
# per-row two-form contrast. Plan: reports/plans/2026-11-19_plan_rl-67m.md (Stage 3b (E223)). Settings, expected values
# and rules: experiments/rl_67m/results/predictions_E223.json (run_contrast.py refuses unless it is committed unchanged;
# it also checks E220's predictions_E220.json, so run on the cluster from a clone, not on a pod with one PRED_SHA).
# One job per argument; submit from the clone root:
#   sbatch --comment='E223 rehearse: contrast ctl' --time=00:20:00 cluster/rl_67m_contrast.sbatch ctl        # first and alone
#   sbatch --comment=E223 --time=00:15:00 --dependency=afterok:<ctl id> cluster/rl_67m_contrast.sbatch 1     # and 2, in parallel
# Times on an L40, each a measured item x its count: model load ~2 min (E222 log 61971); L1's data and reader ~1 min
# (E211 log 61446) and its state <1 min; per pick reader + zero read ~0.6 min (E222 ctl: 18 s zero read, ~20 s rows);
# E53's reader and state ~1.5 min (E222 ctl). ctl ~7 min (2 + 2 + 2 x 0.6 + 1.5). Per state at most 18 s (E222 jobs
# 61972-61973 read 18.1-18.4 s with k_half, none here): jobs 1 and 2 ~7 min each (2 + 0.6 + 13 reads x 18 s).
# Host memory as E222's jobs (peak 6.1 GB).
# Inputs: as cluster/rl_67m_stage3.sbatch (the model, artifacts/sets/*.npy, experiments/rl_67m/results/{stage0,stage1,
# predictions,predictions_E220,stage2b_*}.json, results/large/stage1_samples.pt, results/large/stage2b/*.pt,
# experiments/transport_test/results/large/windows.pt); ctl also E211's L1 (AG News from the HF cache,
# experiments/realistic_tasks/results/{run2_L1.json,large/run2/L1_s0_0.005.pt}) and E53's lowercase inputs (E136/E143
# records, experiments/performed_anchor/results/large/dW_lowercase_s0_step300.pt). No French Wikipedia stream: the
# anchor is not re-read.
# Writes experiments/rl_67m/results/contrast_<job>.json and results/contrast/<behaviour>_<arm>_s<seed>.json.
#SBATCH --job-name=commute-rl67m-s3b
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as E211, E222
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/rl_67m/results/predictions_E223.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/rl_67m/run_contrast.py "$1"
