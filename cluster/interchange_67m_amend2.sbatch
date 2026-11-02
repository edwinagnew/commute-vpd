#!/bin/bash
# E66 Amendment 2: the patch on the saved fine-tunes, read as exploratory, after job 48971 stopped at stage 1. No label, no verdict.
# W0 alone first; then T, T_off, Q at blocks 1-3 on E62's six states, the eight copy and random-map states and E62 Amendment 2's
# six scaled copies. The neuron reading and the controls do not run; the write and sharpening readings are copied from run.json.
# Plan: reports/plans/2026-11-02_plan_e66-interchange-67m.md, "Amendment 2".
# Stops and voids: experiments/interchange_67m/results/predictions_amend2.json (amend2.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/interchange_67m_amend2.sbatch
# Exit 1 at: the transplant check over 1e-4; half the six maps void at every block; R(W0) off job 48971's by over 1e-4 or another
# l*; a state's gain off its filed value by over 0.02. Every number is printed to this log.
# Inputs on NFS (gitignored, read only): experiments/induction_cue/results/large/{windows.pt,{copy,lookup}_s{0,1,2}.pt};
# experiments/regating_table/results/large/{windows_{ident32,ident32b,rand32,rand32b}.pt,{ident32,ident32b,rand32,rand32b}_s{0,1}.pt};
# Pile rows artifacts/sets/train_0_32768.npy (cached); the target model in artifacts/vpd/; committed JSON: interchange_67m
# results/{predictions,run}.json, induction_cue run.json, amend2.json, baseline_neurons.json.
# Writes experiments/interchange_67m/results/amend2.json (refuses if it exists). Saves no state. Deletes nothing, commits nothing.
# Expected about 5 min: job 48971 took 2 min 40 s with heavier work per map; 20 rows at about 3 s each, plus 14 G re-reads and steering.
#SBATCH --job-name=commute-interchange-67m-a2
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
.venv/bin/python experiments/interchange_67m/amend2.py "$@"
