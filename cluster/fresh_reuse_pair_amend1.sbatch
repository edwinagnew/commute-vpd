#!/bin/bash
# E125 Amendment 1 (descriptive, beside the frozen verdict): E125's removal share read at the repeated word's own position
# (blocks 0-2, the probe refit there) and at the trigger's block 2, on the four saved fine-tunes. Plan:
# reports/plans/2026-11-06_plan_e125-fresh-reuse-pair.md, "Amendment 1"; settings and expected values in
# experiments/fresh_reuse_pair/results/predictions_amend1.json (the run refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/fresh_reuse_pair_amend1.sbatch
# Exit 0 done ("AMEND1 DONE"; "STAGE n DONE" per unit: 0 the controls, 1-4 the sites); 1 a stop (a control missed, or
# the random direction over 0.10; reason under "stopped" in results/amend1.json); 3 too little time left (resubmit,
# finished sites are kept): ~/commute-agents/fleet/resubmit --code 3 cluster/fresh_reuse_pair_amend1.sbatch
# Inputs: results/stage0.json, stage1.json, results/large/task.pt, probes.pt, dW_{real,twin}_s{0,1}.pt (sha256 checked),
# the target model in artifacts/vpd/. Writes results/amend1.json (after every unit), results/amend1.out (this log).
# Time on a debug L40: ~4 min (model load ~1 min as E125's stage 0; two residual passes over 9,600 + 1,600 windows; 34
# removal reads of three 1,024-window forward passes each, ~2 s per read from E125 stage 1's reads). Memory: the four
# fine-tunes in fp32 (~0.45 GB) and task.pt; 16G, half of fresh_reuse_pair.sbatch's 32G, which was sized for stage 2's
# gate sets.
#SBATCH --job-name=commute-e125a1
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
rc=0
.venv/bin/python experiments/fresh_reuse_pair/amend1.py 2>&1 | tee -a experiments/fresh_reuse_pair/results/amend1.out || rc=$?
echo "=== FINISHED E125 Amendment 1 (exit $rc) — results in $COMMUTE_REPO/experiments/fresh_reuse_pair/results/amend1.json ==="
exit $rc
