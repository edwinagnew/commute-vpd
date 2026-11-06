#!/bin/bash
# E104 (qwen_d3_gsm8k), the carried D3 rows on the GSM8K arms, on one cluster L40, debug QoS, $0:
# experiments/qwen_d3_gsm8k/run.py. Plan: reports/plans/2026-11-06_plan_e104-d3-gsm8k.md (appendix "Jobs"); rules:
# experiments/qwen_d3_gsm8k/results/predictions.json (the run refuses until the lead files it). Submit from the clone root;
# not submitted by the code writer:
#   sbatch cluster/qwen_d3_gsm8k.sbatch --stage control                                # A: stage 0, alone (~30 min)
#   sbatch cluster/qwen_d3_gsm8k.sbatch --states R_s0_10,R_s0_100,R_s70_10,R_s70_100   # B, after A passed (~45-85 min)
#   sbatch cluster/qwen_d3_gsm8k.sbatch --states N_s0_10,N_s0_100,N_s70_10,N_s70_100   # C, with B (~45-85 min)
#   sbatch cluster/qwen_d3_gsm8k.sbatch --states F_s0_10,F_s0_100,S1,S2,S3             # D, after B (~65-115 min)
# Exit 0 done; 1 a stop (results/stopped.json; every run refuses while it exists) or a refusal; 3 the time left was under
# the longest state so far: resubmit the same command, written states are skipped (fleet/resubmit --code 3).
# Card: E94 peaked at 42.5 GiB; the run stops over 44 GiB. Host: E94 88.4 GB; the run stops over 105 GB, so --mem=110G
# (E94's). Time: per state 9-20 min (E94's run.log), a scaled copy ~3 min more; load and Σ ~6 min a job; debug's 2 h.
#SBATCH --job-name=commute-qwen_d3_gsm8k
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
.venv/bin/python experiments/qwen_d3_gsm8k/run.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
