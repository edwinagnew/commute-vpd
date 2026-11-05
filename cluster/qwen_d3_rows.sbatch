#!/bin/bash
# E94 (qwen_d3_rows), the D3 rows, on one cluster L40, debug QoS, $0: experiments/qwen_d3_rows/run.py. Plan:
# reports/plans/2026-11-05_plan_e94-d3-rows.md; rules: experiments/qwen_d3_rows/results/predictions.json. Submit from the
# clone root, one stage at a time, each after the last ends; not submitted by the code writer:
#   sbatch cluster/qwen_d3_rows.sbatch --stage control                              # stage 0, alone (~25 min)
#   sbatch cluster/qwen_d3_rows.sbatch --states a1_s0,a1_s1                         # stages 1 (unit nulls, ~10-16 min) and 2
#   sbatch cluster/qwen_d3_rows.sbatch --states a2_s42_matched,a2_s142_matched      # stage 3
#   sbatch cluster/qwen_d3_rows.sbatch --states a2_s42_full,a2_s142_full            # stage 4
#   sbatch cluster/qwen_d3_rows.sbatch --states gauss_small,alive_small             # stage 5, after stages 2 and 4
#   sbatch cluster/qwen_d3_rows.sbatch --states gauss_big,alive_big                 # stage 5
#   sbatch cluster/qwen_d3_rows.sbatch --states a2_s42_past,a2_s142_past            # stage 6
# Exit 0 done; 1 a stop (results/stopped.json; every run refuses while it exists) or a refusal; 3 the time left was under
# the longest state so far: resubmit the same command, written states are skipped (fleet/resubmit --code 3).
# Card: E72's reader peaked at 45.8 GB allocated (42.7 GiB); the run stops over 44 GiB. Host: E57 92.7 GB, E68 81 GB with
# Σ, Γ parked and two weight sets; the run stops over 105 GB, so --mem=110G (the brief's).
#SBATCH --job-name=commute-qwen_d3_rows
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
.venv/bin/python experiments/qwen_d3_rows/run.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
