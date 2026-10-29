#!/bin/bash
# E52 Amendment 1: P4 re-asked. W0 prompted with n = 0, 4, 16, 31 demonstrations placed next to the query.
# Decision: reports/decisions/2026-10-29_decision_e52-p4.md (option A). Rules: experiments/baseline_neurons/results/
# predictions_amend1.json.
#   cd ~/commute-agents/<role> && sbatch cluster/baseline_neurons_amend1.sbatch
# Stage 0 (the induction control, then the n = 0 check) runs first and the job exits 1 on a miss. Inputs on NFS
# (gitignored): windows_*.pt in experiments/regating_table/results/large/ (-> ~/commute-data/results-large/
# regating_table/), the target model in artifacts/vpd/; E52's results/baseline_neurons.json (committed) for each state's
# G_score. No delta-W is loaded. Writes experiments/baseline_neurons/results/amend1.json (small) only.
# Expected: 2-4 minutes (model load dominates; 13 forwards of at most 48 x 114 tokens).
#SBATCH --job-name=commute-baseline-neurons-a1
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
test -f experiments/baseline_neurons/results/predictions_amend1.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/baseline_neurons/amend1.py "$@"
