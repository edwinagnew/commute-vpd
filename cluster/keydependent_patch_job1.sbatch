#!/bin/bash
# E56, neuron patching on E54's key-dependent pair and seeds 2-4 of the pair.
# Plan: reports/plans/2026-10-30_plan_baseline-next-draft.md. Rules: experiments/keydependent_patch/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/keydependent_patch_job1.sbatch
# Job 1: E52's neuron patching (B2) on E54's six saved weight changes. Stage 0 (all six: sha256 against E54's
# stage2.json; identity patch 0 and full residual patch 1; r, S and S_lo reproduce stage2.json / amend1.json within 1e-6)
# runs first and the job exits 1 on a miss; then N16 and k50 per state and P1 on two seeds (interim).
# Inputs on NFS (gitignored): experiments/keydependent_elicit/results/large/{windows_elicit,windows_cipher,
# {elicit,cipher,cipher_own}_s{0,1}}.pt (~113 MB each); experiments/transport_test/results/large/alive.pt; the target
# model and decomposition in artifacts/vpd/; E54's stage1.json, stage2.json, amend1.json (committed).
# Writes experiments/keydependent_patch/results/job1.json (small) only; commits nothing: the runner commits job1.json
# and the log (job 2 refuses until job1.json is committed unchanged).
# Expected: 3-6 minutes (E54's stage 2 read r and S on six states in ~1.5 min after the model load; B2 adds ~10
# forwards of at most 48 x 41 tokens per state).
#SBATCH --job-name=commute-keydependent-patch-1
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
.venv/bin/python experiments/keydependent_patch/run.py --job 1 "$@"
