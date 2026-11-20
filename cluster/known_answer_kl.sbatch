#!/bin/bash
# E228 known-answer stage (check 1), cell P2.44: experiments/random_kl_share/known_answer.py.
# Plan reports/plans/2026-11-20_plan_p2-44-random-kl-matched.md ("Known-answer stage (check 1)").
#   sbatch --comment='E228 stage: known-answer items (check 1)' cluster/known_answer_kl.sbatch
# Writes experiments/random_kl_share/results/known_answer/ and known_answer.log. ~5 min on an L40, peak GPU ~7 GB.
#SBATCH --job-name=commute-known-kl
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
L=experiments/random_kl_share/results/known_answer.log
( PRED_SHA=4f096209a8610870b9fb4f1197706d4b249bc936a347e4b8176829ecf4881173 .venv/bin/python experiments/random_kl_share/known_answer.py 2>&1 | tee -a $L; exit ${PIPESTATUS[0]} )
rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/random_kl_share/results/known_answer/ ===" | tee -a $L
exit $rc
