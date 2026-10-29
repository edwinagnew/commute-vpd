#!/bin/bash
# E52, neuron patching (B2), steering (B3) and prompting the original model (B4) on E12's copy and cipher fine-tunes.
# Plan: reports/plans/2026-10-29_plan_baselines-scoping.md, section "B2 + B3 + B4 = E52".
#   cd ~/commute-agents/<role> && sbatch cluster/baseline_neurons.sbatch
# Stage 0 (the controls) runs first and the job exits 1 on a miss. Inputs on NFS (gitignored): E46's states and windows
# in experiments/regating_table/results/large/ (-> ~/commute-data/results-large/regating_table/: {ident32,ident32b,
# rand32,rand32b}_s{0,1}.pt, code_step10.pt, code_held.pt, windows_*.pt), Pile rows artifacts/sets/val_0_64.npy, the
# target model in artifacts/vpd/. Writes experiments/baseline_neurons/results/baseline_neurons.json (small) only.
# Expected: a few minutes (model load and ~0.9 GB of weight changes from NFS dominate; every forward is < 1 s).
#SBATCH --job-name=commute-baseline-neurons
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/baseline_neurons/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/baseline_neurons/run.py "$@"
