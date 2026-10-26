#!/bin/bash
# lora_split: every saved LoRA and a re-run fine-tune through one split table (four blocks, rescaling share, both splits'
# pieces), then part 2 (do-nothing, least-squares and scaled adjoint split states on Pile at steps 10, 100, 1000).
# Design check confirmed by the user 2026-10-20 (reports/plans/2026-10-20_plan_lora-split.md). The arguments go to run.py:
#   cd ~/commute-agents/<role>
#   sbatch --job-name=commute-lora_split-smoke --time=00:25:00 cluster/lora_split.sbatch all --smoke
#   sbatch cluster/lora_split.sbatch all
#   sbatch --job-name=commute-lora_split-pieces --time=00:40:00 cluster/lora_split.sbatch pieces   # table + pieces only, ~15 min
# Before that: `cluster/sync.sh up` from the laptop, predictions filed (results/predictions.json), and the E7 and E9
# factor files under experiments/lora_commute/results/large and experiments/lora_plateau/results/large on the cluster.
# Expected: smoke ~6 min; scored ~60 min (fine-tune to 1000 steps ~4 min, 33 split tables ~8 min on CPU, 21 Pile
# evaluations at ~2 min each).
#SBATCH --job-name=commute-lora_split
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:45:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/lora_split/results/predictions.json || { echo "file the predictions first (results/predictions.json)"; exit 1; }
.venv/bin/python experiments/lora_split/run.py "$@"
