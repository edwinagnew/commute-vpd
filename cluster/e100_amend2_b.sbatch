#!/bin/bash
# E100 Amendment 2, job B: the strong-bigram sharpen map and its shuffled twin, two seeds each (E54's recipe), read by the sum
# of single reverts under the rule job A kept. Plan: reports/plans/2026-11-05_plan_e100-amend2.md. Rules:
# experiments/fed_or_written/results/predictions_amend2.json. amend2_sharpen.py refuses unless the predictions are committed
# unchanged and results/amend2.json records a kept rule (job A's stage 1).
#   cd ~/commute-agents/<role> && sbatch cluster/e100_amend2_b.sbatch
# Stages: 0 (W0) the strong keys (stop under 24), the two tables, the zero update reads 0; 1 sharpen s0, s1; 2 twin s0, s1
# (stop unless ||dW|| within 5% of the sharpen's; every state stops on a step-1 gain not finite and positive). Each prints
# "STAGE <n> DONE". A resubmit skips stage 0 when its windows match their recorded sha256, and every state already read.
# Writes experiments/fed_or_written/results/amend2_sharpen.json, amend2_sharpen.out and results/large/ (windows and four
# bf16 dW, ~57 MB each). Deletes nothing, commits nothing.
# Expected 8-10 min on a debug L40 (predictions_amend2.json cost.job_b: four 300-step fine-tunes at ~25-40 s each from E58
# job 46332; the vocabulary pass over 32,768 Pile rows unmeasured).
#SBATCH --job-name=commute-e100-amend2-b
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/fed_or_written/amend2_sharpen.py 2>&1 | tee -a experiments/fed_or_written/results/amend2_sharpen.out || rc=$?
echo "=== FINISHED E100 Amendment 2 job B (exit $rc) — results in $COMMUTE_REPO/experiments/fed_or_written/results/ ==="
exit $rc
