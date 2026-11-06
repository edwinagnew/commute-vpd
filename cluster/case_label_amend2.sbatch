#!/bin/bash
# E112 Amendment 2, the last (decisions.md row "E112 Amendment 2"): stage 2's P3 re-read on the four saved states with
# full linear concept erasure (INLP of both case labels per block, merged) at every position of blocks l*..3; do-nothing
# (same-rank random basis per block) first, exit 1 if over 0.10.
# Rules: experiments/case_label/results/predictions.json (unchanged) and predictions_amend2.json (amend2.py refuses
# unless both and predictions_amend1.json are committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/case_label_amend2.sbatch
# Exit 0 done; 1 the do-nothing removal over 0.10 on a state (no state read), or a saved state's sha256 differs.
# Inputs: results/large/{task,probes}.pt and dW_{real,twin}_s{0,1}.pt (stages 0-1), results/amend1.json, artifacts/vpd/.
# Writes experiments/case_label/results/amend2.json (after every step), results/large/amend2_bases.pt (< 2 MB) and
# amend2.out (this log, appended).
# Expected time on a debug L40 (estimate): about 6 min (model load ~1; residuals of 12,800 windows at two positions and
# at most 8 x 65 float64 ridge fits on the GPU < 1; refits at block 3 < 1; W0 reads under 3 removals and 4 x 3 shares
# of 1,024 windows each ~3). Memory: Amendment 1 of the same code; stage 2 peaked at 6.1 GB host; 16G.
#SBATCH --job-name=commute-e112-a2
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
.venv/bin/python experiments/case_label/amend2.py 2>&1 | tee -a experiments/case_label/results/amend2.out || rc=$?
echo "=== FINISHED E112 amend2 (exit $rc) — results in $COMMUTE_REPO/experiments/case_label/results/ ==="
exit $rc
