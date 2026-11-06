#!/bin/bash
# E122: the damage-crossed pair. Plan reports/plans/2026-11-06_plan_e122-damage-crossed-pair.md.
#   cd ~/commute-agents/<role>                      # a clone with the results/large symlinks (fleet/link-data <role>), not a worktree
#   ~/commute-agents/fleet/resubmit --code 3 cluster/damage_crossed.sbatch   # ~50 min of work in 2-3 jobs; exit 3 = more to do
# Before: experiments/damage_crossed/results/predictions.json committed. Inputs on NFS: transport_test/results/windows.json,
# transport_test/results/large/{alive,windows}.pt, broad_fragility/results/large/{trained_step1000,a2_tt_lora_s0}.pt,
# artifacts/sets/{train_0_4096,train_0_32768,val_0_256,val_256_256}.npy.
# Writes results/damage_crossed.json and results/large/{ft_adam300,ft_c1,ft_c2}.pt (fp32, ~113 MB each),
# {tt_recipe,tt_c1,tt_c2}.pt (bf16 offsets, ~60 MB each), maps/<state>_<set>.pt (~0.5 MB each) and run.log.
# A rerun continues from those files. Exit 1: a stop; the reason is `.stopped` in the file (move it aside to start again).
#SBATCH --job-name=commute-e122
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
[ $# -eq 0 ] || { echo "damage_crossed.sbatch: takes no arguments" >&2; exit 1; }
LOGS=experiments/damage_crossed/results/large
mkdir -p "$LOGS"
rc=0
.venv/bin/python experiments/damage_crossed/run.py 2>&1 | tee -a "$LOGS/run.log" || rc=$?
echo "=== FINISHED (exit $rc) — results in $COMMUTE_REPO/experiments/damage_crossed/results/damage_crossed.json and $COMMUTE_REPO/$LOGS/ ==="
exit $rc
