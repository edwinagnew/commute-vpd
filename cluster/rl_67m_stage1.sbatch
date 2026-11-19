#!/bin/bash
# E215 (cell P2.41 stage 1): what the 67M base model already samples, per behaviour with a programmed reward. Plan:
# reports/plans/2026-11-19_plan_rl-67m.md. Rule and bars: experiments/rl_67m/results/predictions.json (run_stage1.py
# refuses unless it is committed unchanged).
#   cd <clone> && ~/commute-agents/fleet/link-data <role or .> && sbatch --comment=E215 cluster/rl_67m_stage1.sbatch
# Stage 0 (KL read at a zero and a random update, prompts, reward on the texts' own continuations; alone, stops on a
# miss), stage 1 (13 behaviours x 256 prompts x 8 samples of 16 tokens, then the filed rule); each prints
# "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1.
# Inputs: the model in artifacts/vpd/; cached Pile val rows artifacts/sets/val_0_8192.npy (and val_0_64.npy or any
# file holding rows 0-63); E1's emoticon windows experiments/transport_test/results/large/windows.pt; downloads inside
# the job: the GPT-NeoX tokenizer (cached), French and German Wikipedia streamed (first ~300 articles each).
# Writes experiments/rl_67m/results/stage{0,1}.json and results/large/stage1_samples.pt (~3 MB).
# Expected on an L40: about 10 min (model load 2; prompts and Wikipedia 2; sampling 104 batches of 256 sequences x 16
# tokens, measured by the run). Host memory: the model and 8128 cached rows, under 10 GB.
#SBATCH --job-name=commute-rl67m-s1
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
test -f experiments/rl_67m/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/rl_67m/run_stage1.py
