#!/bin/bash
# E56, neuron patching on E54's key-dependent pair and seeds 2-4 of the pair.
# Plan: reports/plans/2026-10-30_plan_baseline-next-draft.md. Rules: experiments/keydependent_patch/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/keydependent_patch_job2.sbatch
# Job 2: seeds 2, 3, 4 of E54's elicit and cipher fine-tunes (cipher read at the elicit update norm and at its own
# step), then r, S, S_lo, N16 and k50 on all nine states, then P1 and P2 on five seeds and P3 on seeds 2-4.
# Refuses unless job1.json passed stage 0 with P1 holding on two seeds, and job1.json and predictions.json are committed
# unchanged. Stage 0 (windows sha256, keys, values and W0's elicit rate against stage1.json; seed-0 elicit r against
# stage2.json, within 1e-6) runs before any fine-tune and exits 1 on a miss.
# Inputs as job 1. Writes experiments/keydependent_patch/results/job2.json (small) and six weight changes (~113 MB each)
# to experiments/keydependent_patch/results/large/{elicit,cipher}_s{2,3,4}.pt (a link into ~/commute-data/results-large);
# commits nothing.
# Expected: 6-10 minutes. Measured from E54's job 45974 on a debug L40: a 300-step elicit fine-tune 11 s, a cipher
# fine-tune with 32 checkpoints 16 s, r and S on six states ~1.5 min; here three seeds of (two fine-tunes, two 113 MB
# saves, three full reads) plus the model load. One job fits the 30-minute cap, so job 2 is not split.
#SBATCH --job-name=commute-keydependent-patch-2
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
.venv/bin/python experiments/keydependent_patch/run.py --job 2 "$@"
