#!/bin/bash
# P2.47 (E230): the re-gating share with VPD's subcomponents kept and Gamma replaced by a top-k gate on each
# subcomponent's own inner activation, on the 68 saved fine-tunes of the nine built pairs. Plan
# reports/plans/2026-11-20_plan_p2-47-topk-gate.md. Rules: experiments/topk_gate/results/predictions.json (run.py refuses
# unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch --comment=E230 cluster/topk_gate.sbatch read
#   (score again with no GPU: ~/commute-agents/fleet/cpu-run python3 experiments/topk_gate/run.py score)
# Stage 0 (the zero change, alone; exit 1 on a miss) writes results/zero.json; stage 1 writes results/states/<state>.json
# as each state ends (a rerun skips finished states; a gamma-gate miss on the saved share exits 1) and per-row cells to
# results/large/cells/; then results/score.json. The runner waits on `STAGE n DONE`.
# Inputs on NFS (gitignored): experiments/topk_gate/results/manifest.txt (`run.py manifest`).
# Expected on a debug L40 (estimate from E136, 68 states in ~5 min, job 56944, at ~2.5x the forwards): ~12-15 min with
# model load; projected-time stop at 45 min after 4 states. No gate sets are held across batches: host memory ~10 GB.
#SBATCH --job-name=commute-topk-gate
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as E222/E228
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/topk_gate/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/topk_gate/run.py "$@"
