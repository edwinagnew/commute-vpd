#!/bin/bash
# E104 Amendment 1 (qwen_d3_gsm8k), S1 again by a bisection on all 128 maths rows, one cluster L40, debug QoS, $0:
# experiments/qwen_d3_gsm8k/amend1.py. Plan: reports/plans/2026-11-06_plan_e104-d3-gsm8k.md, "Amendment 1"; rule:
# experiments/qwen_d3_gsm8k/results/predictions_amend1.json (the run refuses unless it is committed unchanged). Submit from
# the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_d3_gsm8k_amend1.sbatch
# Exit 0 done (results/amend1.json; rerunning after S1a1 is written only rescores); 1 a stop (results/stopped.json) or a
# refusal. Time, from E104's run.log: load 48-131 s; a try = base cells on 128 rows without counts, 109-140 s; the
# control try plus at most 8 bisection tries; the read 10-14 min (cells 2 min, split 405-759 s). Expected ~30 min, worst
# ~42 min. Card and host as E104 (peak 42.1 GiB, 80.5 GB; stops over 44 GiB, 105 GB), so --mem=110G.
#SBATCH --job-name=commute-qwen_d3_gsm8k_amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_d3_gsm8k/amend1.py --decomp "$COMMUTE_QWEN_DIR" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
