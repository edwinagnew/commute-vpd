#!/bin/bash
# E139 Amendments 1 and 2 (cell P1.4). Page: reports/predictions/2026-11-10_prediction_e139-amendments-1-2.md. Filed rules:
# experiments/reuse_variable_finetune/results/predictions_amend{1,2}.json (each run refuses unless its file is committed unchanged).
# One L40, debug QoS, $0. Submit from the clone root, Amendment 1 first (it reads E139's saved fine-tunes, ~6 min):
#   sbatch --comment='E139 amend1' cluster/reuse_variable_finetune_amend.sbatch amend1
#   sbatch --comment='E139 amend2' cluster/reuse_variable_finetune_amend.sbatch amend2 0     (and amend2 1, amend2 2: one job per seed, ~80 min each, side by side)
# amend2 exit 3 means the wall limit (105 min) was reached with results saved: resubmit the same line, it resumes.
# Inputs: E139's results/control.json, always_on_s*.json, large/always_on_s*.pt; E137's stage0.json, large/pieces.pt.
# Writes: results/amend1.json; results/amend2_always_on_s<seed>.json, results/large/amend2_always_on_s<seed>_step{300,1000,3000}.pt (268 MB each).
# Memory, from E139's logs (jobs 57673, 57675): host 6.1 GB peak; card about 35 GB at the H `final` site, as E137.
#SBATCH --job-name=commute-r2-amend
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:55:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --comment=E139

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
script="$1"; shift
case "$script" in amend1|amend2) ;; *) echo "first argument: amend1 or amend2" >&2; exit 2 ;; esac
.venv/bin/python "experiments/reuse_variable_finetune/$script.py" "$@"
