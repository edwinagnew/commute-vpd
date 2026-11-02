#!/bin/bash
# E66, the feasibility probe: can a context patch and a neuron ablation read anything on the 67M's fine-tunes? W0, three built
# controls, E62's six states, the eight copy and random-map states, E62 Amendment 2's scaled copies. The one training is the
# sharpening control (30 steps). Plan: reports/plans/2026-11-02_plan_e66-interchange-67m.md, "Amendment 1: feasibility probe".
# Rules: experiments/interchange_67m/results/predictions.json (run.py refuses unless it is committed unchanged). Controls only;
# readings on saved states are exploratory.
#   cd ~/commute-agents/<role> && sbatch cluster/interchange_67m.sbatch
# Exit 1 at: stage 0, W0 alone (the identity patch or zero update over 1e-4; half the six maps void at every block); stage 1,
# amplify's T not over the write's by 3 s on half the readable maps; stages 2-3, a state's gain off its filed value by over 0.02.
# Every scored number is printed to this log.
# Inputs on NFS (gitignored, read only): experiments/induction_cue/results/large/{windows.pt,{copy,lookup}_s{0,1,2}.pt};
# experiments/regating_table/results/large/{windows_{ident32,ident32b,rand32,rand32b}.pt,{ident32,ident32b,rand32,rand32b}_s{0,1}.pt};
# experiments/keydependent_elicit/results/large/windows_elicit.pt; Pile rows artifacts/sets/train_0_32768.npy (cached); the target
# model in artifacts/vpd/; committed JSON: induction_cue run.json, amend2.json, baseline_neurons.json, keydependent_elicit
# stage2.json, regating_truth_table stage1.json.
# Writes experiments/interchange_67m/results/run.json (refuses if it exists). Saves no state. Deletes nothing, commits nothing.
# Expected: 10-15 min, unmeasured: 6 maps' pools ~1, stage 0 ~2, stage 1 ~4, 20 state reads at ~15-25 s each. 30-min wall.
#SBATCH --job-name=commute-interchange-67m
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
.venv/bin/python experiments/interchange_67m/run.py "$@"
