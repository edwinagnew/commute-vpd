#!/bin/bash
# E102 (qwen_carrying) stage A on one cluster L40, debug QoS, $0: experiments/qwen_carrying/run.py. Rules:
# experiments/qwen_carrying/results/predictions.json. Two jobs, the second after the first ends with control (i) passed;
# submit from the clone root:
#   sbatch cluster/qwen_carrying.sbatch --states gauss,a1_s0,a1_s1
#   sbatch cluster/qwen_carrying.sbatch --states a2_s42_matched,a2_s142_matched,a2_s42_full,a2_s142_full
# Time, per plan "Cost": job 1 about 19 min (W0 stage about 10 incl. the FineWeb fetch, E94's Gaussian about 7, anchor 1
# about 2.5), job 2 about 10 min. Built on experiments/qwen_comparators/results/run.log:63-64 (a TOFU state read, 32,768
# tokens, in 5.8 s on an L40), experiments/qwen_size_grid/results/run.log:5-6 (24 s to load a 5.64 GB state over NFS) and
# experiments/qwen_d3_rows/results/run.log:17 (Σ on 196 sites in 181 s). The run pauses with exit 3 before a unit that
# would pass 27 min (settings.sizes.budget_s); resubmit the same command and it resumes from what is written.
# --mem: E92's job peaked at 11.0 GB host (qwen_comparators/results/run.log:75); the Gaussian adds Σ (11.3 GB fp64),
# anchor 1's δW (5.6 GB) and the bf16 direction (2.8 GB). Settings otherwise as cluster/qwen_comparators.sbatch.
# Exit 0 done; 1 a stop (results/stopped.json) or a refusal; 3 the time budget: resubmit.
#SBATCH --job-name=commute-qwen_carrying
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
.venv/bin/python experiments/qwen_carrying/run.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
