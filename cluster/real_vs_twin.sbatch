#!/bin/bash
# E140 (cell P2.4): do steering or the neuron version tell a real fine-tune from its shuffled-label twin, where the share
# cannot? Plan: reports/plans/2026-11-10_plan_e140-real-vs-twin.md. Rules: experiments/real_vs_twin/results/predictions.json
# (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch cluster/real_vs_twin.sbatch
# Stages 0-3 in one job; each prints "STAGE n DONE" after its results file. A filed stop prints "STOP: ..." and exits 1.
# Inputs on NFS (gitignored): experiments/{case_label,fresh_reuse_pair}/results/large/{task.pt,dW_{real,twin}_s{0,1}.pt},
# the target model and decomposition in artifacts/vpd/. Writes experiments/real_vs_twin/results/stage{0,1,2,3}.json and the
# four seed-2 updates to experiments/real_vs_twin/results/large/dW_{case,repeat}_{real,twin}_s2.pt (bf16, ~57 MB each).
# Expected on an L40: about 10 min. Model load ~2 min; stage 0 ~1 min; stage 1 ~1.5 min (five fine-tunes; E112's four took
# 40 s, experiments/case_label/results/stage1.out); stage 2 ~1 min (12 share reads on 256 rows; E135's took ~9 s each);
# stage 3 ~5 min (12 updates: vectors on 9,600 windows, a 48-setting grid on 512, reads on 256). E135, the nearest job,
# took 3.5 min with a peak host RSS of 6.8 GB (cluster/logs/commute-steering-copy-lookup-56927.out).
#SBATCH --job-name=commute-real-vs-twin
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --comment=E140
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/real_vs_twin/results/predictions.json || { echo "file the predictions first"; exit 1; }
test -L experiments/real_vs_twin/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data compare-lead"; exit 1; }
.venv/bin/python experiments/real_vs_twin/run.py
