#!/bin/bash
# E93 Amendment 1 (qwen_d1_table): the scaled adjoint split on D1's first table, one cluster L40, $0:
# experiments/qwen_d1_table/amend1.py. Plan: reports/plans/2026-11-05_plan_e93-d1-table.md, "Amendment 1"; rules:
# experiments/qwen_d1_table/results/predictions_amend1.json. Submit from the clone root, one job at a time, each after
# the last ends; not submitted by the code writer:
#   sbatch --qos=debug cluster/qwen_d1_table_amend1.sbatch --stage control                           # ~22 min, alone
#   sbatch cluster/qwen_d1_table_amend1.sbatch --states a1_s0,a1_s1,a2_s42,a2_s142                   # ~45 min
#   sbatch cluster/qwen_d1_table_amend1.sbatch --states u1,u2,R_s0,R_s70,N_s0,N_s70,F_s0             # ~75 min
# Exit 0 done; 1 a stop (results/amend1/stopped.json; every run refuses while it exists) or a refusal; 3 the time left
# was under the longest state so far: resubmit the same command (fleet/resubmit --code 3).
# Card: E93 peaked at 42.7 GiB with the same passes; the run stops over 44 GiB. Host: E93's 95.6 GB less its random,
# signs and least-squares X, plus Σ (~11 GB, E94's 72 GB with Σ and Γ parked); the run stops over 105 GB.
#SBATCH --job-name=commute-qwen_d1_table_amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_d1_table/amend1.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
