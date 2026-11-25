#!/bin/bash
# E312 (cell P2.84): Jain et al. (2023)'s reverse fine-tune on Fig 4's 48 states, a descriptive per-task panel for Fig 6.
# Card experiments/reverse_ft_fig4/card.md; rules and expected values experiments/reverse_ft_fig4/results/predictions.json
# (run.py refuses unless it is committed unchanged; on a pod set PRED_SHA).
# Stage 0 first, alone; then the items as an array of 4 shards (run.py stage 0 prints the shard count it projects):
#   cd ~/commute-agents/baseline-lead && ~/commute-agents/fleet/link-data baseline-lead
#   sbatch --comment=E312 cluster/reverse_ft_fig4.sbatch 0
#   sbatch --comment=E312 --dependency=afterok:<stage-0 job id> --array=0-3 cluster/reverse_ft_fig4.sbatch items 4
#   then, no GPU: ~/commute-agents/fleet/cpu-run python3 experiments/reverse_ft_fig4/run.py --stage summary
# Stage 0: the zero update, the kappa floor, C3's reproductions, the planted-pair gate, the gain fold, the 3-state null
# probe and projection, the probe states under Pile cross-entropy; prints "STAGE 0 DONE", or "STOP: ..." and exits 1.
# Items: 111 reverse fine-tunes (48 states, 48 rotated nulls, 15 gain-only), each written to results/items/ as it lands;
# a rerun skips finished items.
# Expected on a debug L40, from measured runs: C3 read a state (pruning + reverse to D/2, 9-19 steps) in 1.3-1.8 s and
# the ladder in 3.0 s (open_pair_instruments/results/{run,ladder}.json .seconds); this run goes on to 0.1 D (about 3x
# the steps) and drops pruning, so about 3-6 s an uncensored item, 0.1-0.2 s a step. Stage 0: model load 1-2 min, ~20
# reverse runs, the e53 windows build (~1 min), Pile CE on 3 states (up to 1000 steps of 32 x 512 tokens each, worst
# ~15 min): about 10-25 min. Items: ~28 per shard, ~3-5 min uncensored; worst case 12 censored nulls x 1000 steps x
# 0.2 s = 40 min a shard. Limit 100 min.
# Memory: the 67M and one task's reference read; e53's read cut to 64 tokens; GPU under 20 GB, host under 16 GB
# (E147/E143 on the same model and windows peaked at 6.1 GB host, 12.1 GB GPU).
#SBATCH --job-name=commute-reverse-ft-fig4
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:40:00
#SBATCH --comment=E312
#SBATCH --output=cluster/logs/%x-%A_%a.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/reverse_ft_fig4/results/predictions.json || { echo "file the predictions first"; exit 1; }
if [ "${1:-0}" = "0" ]; then
  .venv/bin/python experiments/reverse_ft_fig4/run.py --stage 0
else
  .venv/bin/python experiments/reverse_ft_fig4/run.py --stage items --shard "${SLURM_ARRAY_TASK_ID:-0}" --of "${2:-4}"
fi
