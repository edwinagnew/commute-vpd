#!/bin/bash
# E93 (qwen_d1_table), D1's first table, on one cluster L40, debug QoS, $0: experiments/qwen_d1_table/run.py. Plan:
# reports/plans/2026-11-05_plan_e93-d1-table.md; rules: experiments/qwen_d1_table/results/predictions.json. Submit from
# the clone root, one job at a time, each after the last ends; not submitted by the code writer:
#   sbatch cluster/qwen_d1_table.sbatch --stage control                        # stage 0, alone (~25 min)
#   sbatch cluster/qwen_d1_table.sbatch --states a2_s42,a2_s142                # anchor 2, graded (~35 min a state)
#   sbatch cluster/qwen_d1_table.sbatch --states a1_s0,a1_s1,u1,u2             # ~15, 15, 27, 22 min
#   sbatch cluster/qwen_d1_table.sbatch --states R_s0,R_s70,N_s0,N_s70,F_s0    # ~22 min a state
# Exit 0 done; 1 a stop (results/stopped.json; every run refuses while it exists) or a refusal; 3 the time left was under
# the longest state so far: resubmit the same command, written states are skipped (fleet/resubmit --code 3).
# Card: E94's stage 0 peaked at 42.5 GiB with the same reads; the run stops over 44 GiB. Host: no Σ here; Γ parked
# (17 GB), up to four weight sets and the split's X and moved part pinned (5.6 GB each); the run stops over 105 GB.
#SBATCH --job-name=commute-qwen_d1_table
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
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
.venv/bin/python experiments/qwen_d1_table/run.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
