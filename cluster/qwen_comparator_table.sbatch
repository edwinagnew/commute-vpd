#!/bin/bash
# E118 (qwen_comparator_table) on one cluster L40, debug QoS, $0: experiments/qwen_comparator_table/run.py. Controls:
# experiments/qwen_comparator_table/results/predictions.json (committed first: the run refuses otherwise). Two jobs, one
# per part; they write separate files and may run at the same time. Submit from the clone root:
#   sbatch cluster/qwen_comparator_table.sbatch --part compare
#   sbatch cluster/qwen_comparator_table.sbatch --part carry
# Time. compare: about 12 min (model and rows 1.5; C0 and C1 about 2.5, anchor 1's state 50 s in
# qwen_comparators/results/run.log:11; u1 and u2 on the 103 gold rows about 110 s each, R_s0:100's 86 s plus a 24 s load,
# run.log:35 and qwen_size_grid/results/run.log:5-6; six known and wrong states on about 30-40 rows each, about 60 s).
# carry: about 21 min (C0 and C2 about 1.5, E102's anchor 1 state 40 s in qwen_carrying/results/run.log:22-24; the gold
# rows' engaged values under 1 min; 14 states at E102's 82 s per 128-row state with the load, run.log:33-42, the real
# rows' two about 30 s). The run pauses with exit 3 before a unit that would pass 27 min (settings.budget_s); resubmit
# the same command and it resumes from what is written.
# --mem as cluster/qwen_carrying.sbatch (E102 peaked at 23.3 GB host with its Gaussian, 11.0 GB without).
# Exit 0 done; 1 a stop (results/stopped.json) or a refusal; 3 the time budget: resubmit.
#SBATCH --job-name=commute-qwen_comparator_table
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
set +e
.venv/bin/python experiments/qwen_comparator_table/run.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
