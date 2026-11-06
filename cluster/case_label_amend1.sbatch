#!/bin/bash
# E112 Amendment 1 (decisions.md row "E112 Amendment 1"): stage 2's P3 re-read on the four saved states with the rank-2
# case basis removed at every position of every block from l* on; do-nothing (random rank-2) first, exit 1 if over 0.10.
# Rules: experiments/case_label/results/predictions.json (unchanged) and predictions_amend1.json (amend1.py refuses
# unless both are committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/case_label_amend1.sbatch
# Exit 0 done; 1 the do-nothing removal over 0.10 on a state (no state read), or a saved state's sha256 differs.
# Inputs: results/large/{task,probes}.pt and dW_{real,twin}_s{0,1}.pt (stages 0-1), the target model in artifacts/vpd/.
# Writes experiments/case_label/results/amend1.json (after every step) and amend1.out (this log, appended).
# Expected time on a debug L40 (estimate): about 4 min (model load ~1, 11,200 residual windows and 4 ridge fits < 1,
# W0 reads and 4 x 3 shares of 1,024 windows each ~2). Memory: stage 2 of the same code peaked at 6.1 GB host; 16G.
#SBATCH --job-name=commute-e112-a1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/case_label/amend1.py 2>&1 | tee -a experiments/case_label/results/amend1.out || rc=$?
echo "=== FINISHED E112 amend1 (exit $rc) — results in $COMMUTE_REPO/experiments/case_label/results/ ==="
exit $rc
