#!/bin/bash
# E58, the open pair: E54's elicit map and matched cipher fine-tuned on 24 of 32 keys (seeds 0-2), labelled by whether
# the fine-tune carries to the 8 held-out keys, then r, the stand-in ratio and S. Plan:
# reports/plans/2026-10-30_plan_e58-open-pair.md. Rules: experiments/open_pair/results/predictions.json (run.py refuses
# unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/open_pair.sbatch
# One job: stage 0 alone first (E54 file hashes, zero update exactly 0, E54 elicit_s0's r / ratio / S reproduced), exit 1
# on a miss before any fine-tune; then six fine-tunes with labels (exit 1 if elicit_s0's trained-key gain is under 8, or
# if a cipher reads "reuses"); then the readings and the score.
# Inputs (read): experiments/keydependent_elicit/results/{stage1.json,large/windows_{elicit,cipher}.pt,large/elicit_s0.pt},
# experiments/transport_test/results/large/alive.pt, the VPD artifacts.
# Writes experiments/open_pair/results/run.json (small) and, in experiments/open_pair/results/large/ (a link into
# ~/commute-data/results-large/open_pair, made here if missing): four windows files (~0.5 MB) and six weight changes
# {elicit,cipher}_s{0,1,2}.pt (~113 MB each); the log tees to results/large/run.out. Commits nothing.
# Expected 5-10 min on a debug L40 (E56 job 2 on the same card: a 300-step elicit fine-tune 11 s, a cipher fine-tune with
# 32 checkpoints 16 s, r and S on six states ~1.5 min); host RAM peak ~5 GB (32 cipher checkpoints of 113 MB).
#SBATCH --job-name=commute-e58-open-pair
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/open_pair/results/predictions.json || { echo "file the predictions first" >&2; exit 1; }
large=experiments/open_pair/results/large
if [ ! -e "$large" ]; then
  mkdir -p "$HOME/commute-data/results-large/open_pair"
  ln -s "$HOME/commute-data/results-large/open_pair" "$large"
fi
rc=0
.venv/bin/python experiments/open_pair/run.py "$@" 2>&1 | tee "$large/run.out" || rc=$?
echo "=== FINISHED E58 (exit $rc) — results in $COMMUTE_REPO/experiments/open_pair/results/ ==="
exit $rc
