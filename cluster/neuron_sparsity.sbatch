#!/bin/bash
# E63, do fine-tunes built to elicit sit on fewer MLP neurons than fine-tunes built to teach something new?
# Plan: reports/plans/2026-11-02_plan_e63-neuron-sparsity.md. Rules: experiments/neuron_sparsity/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/neuron_sparsity.sbatch
# sha256 of every state and windows file against its source's record, before the model loads. Stage 0 (synthetic
# controls; identity patch per task; the 16-neuron update's A1 and C1; E56's N16 and full-width share on elicit_s0 and
# cipher_s0) runs alone and the job exits 1 on a miss. Then W1 and W2, W1g, A1 and C1 on 26 saved states, and the score.
# Inputs on NFS (gitignored): experiments/{keydependent_elicit,keydependent_patch,regating_table,performed_anchor}/
# results/large/; the target model and decomposition in artifacts/vpd/; Pile rows in artifacts/sets/; the JSON records of
# those runs (committed).
# Writes experiments/neuron_sparsity/results/{stage0,weights,gradients,windows,score}.json (small) only; commits nothing.
# Expected: 6-10 minutes (E56 read six states at 8 grid points in 3-6 min, most of it the model load and the r split;
# here 26 states at 15 grid points, 20 of them 48 windows of 41 tokens, 6 of them 250 windows of 127 tokens, plus code_step10's W1 and W2).
#SBATCH --job-name=commute-neuron-sparsity
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/neuron_sparsity/run.py "$@"
