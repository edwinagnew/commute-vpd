#!/bin/bash
# E85 Amendment 2, does Γ's bf16 rounding of V matter, on one cluster L40, debug QoS, $0: experiments/qwen_d1_teeth/amend2.py. Plan: reports/plans/2026-11-04_plan_e85-adversary-teeth.md, "Amendment 2".
# Submit from the clone root; not submitted by the code writer: sbatch cluster/qwen_d1_teeth_amend2.sbatch
# Exit 1: a stop (results/stopped_amend2.json; a rerun refuses while it exists) or a crash. A rerun starts over; the X on /ephemeral stay.
# Expected 5-10 min. GPU ~40 GB of 46 (amend2.py's docstring). Host 73.9 GB at its peak (job 51785); --mem is that and a margin.
#SBATCH --job-name=commute-qwen_d1_teeth_amend2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:25:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
df -h /ephemeral
set +e
.venv/bin/python experiments/qwen_d1_teeth/amend2.py --decomp "$COMMUTE_QWEN_DIR" --inputs "$HOME/commute-data/results-large/qwen_sft/nowarmup/gstats" --work "/ephemeral/$USER/e85_amend2"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
