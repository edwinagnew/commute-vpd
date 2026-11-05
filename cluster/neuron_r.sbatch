#!/bin/bash
# E98, a neuron analogue of VPD's r on the 67M (r_half headline, r_n, d_n, i_n, e_n), on the method table's rows.
# Plan: reports/plans/2026-11-05_plan_e98-neuron-r.md. Rules: experiments/neuron_r/results/predictions.json (run.py
# refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/neuron_r.sbatch
# sha256 of every state and windows file before the model loads. Stage 0 alone (W0 on every task, E56's full-width share
# reproduced, code_step10 as do-nothing, two built controls fine-tuned on E54's elicit task) and the job exits 1 on a miss.
# Then the cells of 56 saved states, the carrying set of each scored state, the score.
# Inputs on NFS (gitignored): experiments/{regating_table,keydependent_elicit,keydependent_patch,performed_anchor,
# open_pair,open_pair_instruments,induction_cue}/results/large/; the target model in artifacts/vpd/; Pile rows in
# artifacts/sets/; those runs' JSON records (committed).
# Writes experiments/neuron_r/results/{stage0,cells,carrying,score}.json (small) only; commits nothing.
# Expected: 5-10 minutes (E63 read 26 states with 30 patched forwards per batch in 2 min 12 s, job 48527; here 56 states
# at 7 forwards per batch, about 35 scored states at 18 forwards and one backward per batch, two 300-step fine-tunes).
#SBATCH --job-name=commute-neuron-r
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/neuron_r/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/neuron_r/run.py "$@"
