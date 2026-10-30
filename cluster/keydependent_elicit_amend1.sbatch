#!/bin/bash
# E54 Amendment 1: steering's share S read in log-odds (S_lo), the route split's unit, on E54's six saved weight changes.
# Plan: reports/plans/2026-10-29_plan_e54-keydependent-elicit.md, "Amendment 1 (2026-10-30)". Rules:
# experiments/keydependent_elicit/results/predictions_amend1.json (amend1.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/keydependent_elicit_amend1.sbatch
# Stage 0 (all six states: sha256, B3's block and S against stage2.json within 1e-6) runs first and the job exits 1 on a
# miss. Inputs on NFS (gitignored): experiments/keydependent_elicit/results/large/{windows_elicit,windows_cipher,
# elicit_s0,elicit_s1,cipher_s0,cipher_s1,cipher_own_s0,cipher_own_s1}.pt (~113 MB each); the target model in
# artifacts/vpd/; results/stage2.json (committed). Writes experiments/keydependent_elicit/results/amend1.json (small)
# only and commits nothing: the runner commits amend1.json and the log.
# Expected: 2-4 minutes (model load and six sha256 reads dominate; ~20 forwards of at most 97 x 41 tokens per state).
#SBATCH --job-name=commute-keydependent-elicit-a1
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
test -f experiments/keydependent_elicit/results/predictions_amend1.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/keydependent_elicit/amend1.py "$@"
