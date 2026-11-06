#!/bin/bash
# E94 Amendment 1 (qwen_d3_rows), the known-facts states (E103 Amendments 1 and 2) through E94's reads, on one cluster
# L40, debug QoS, $0: experiments/qwen_d3_rows/amend1.py. Plan: reports/plans/2026-11-05_plan_e94-d3-rows.md,
# "Amendment 1"; rules: experiments/qwen_d3_rows/results/predictions_amend1.json. Submit from the clone root, the second
# after the first ends; not submitted by the code writer:
#   sbatch cluster/qwen_d3_rows_amend1.sbatch --states known_s42_matched,known_s142_matched   # W0 on the known-facts rows first (~2 min), then 2 states (~60 min)
#   sbatch cluster/qwen_d3_rows_amend1.sbatch --states known_s42_end,known_s142_end           # ~60 min
# Exit 0 done; 1 a stop (results/stopped.json; every E94 run refuses while it exists) or a refusal; 3 the time left was
# under the longest state so far: resubmit the same command, written states are skipped (fleet/resubmit --code 3).
# Time: E94's anchor-2 states took 20-26 min each on the L40 (results/run.log, "done in" 1194-1577 s) with its 128 TOFU
# rows; the known-facts rows add three passes on ~40 rows (~3 min); load and Σ ~5 min. Card and host as E94's
# (peaks 42.5 GiB, 88 GB): --mem=110G, the run stops over 44 GiB and 105 GB.
#SBATCH --job-name=commute-qwen_d3_rows_amend1
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
.venv/bin/python experiments/qwen_d3_rows/amend1.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
