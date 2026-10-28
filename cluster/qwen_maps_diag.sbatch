#!/bin/bash
# E44 stored_error diagnostic ($0) on one cluster L40, debug QoS: experiments/qwen_maps/diag_stored.py over all 196
# sites for E45's gold-arm δW at steps 2 and 100 (bf16, the same file layout as E40's dW). Stage S stopped on
# layers.2.mlp.up_proj (3.82e-3 > 1e-3); layer 2 runs first. Submit from the clone root; not submitted by the code writer:
#   sbatch cluster/qwen_maps_diag.sbatch
# The output is appended per (site, δW) and the job resumes from it: if it hits the 30-min limit before
# "all sites done", submit the same command again. Host memory: Σ of every site (11.3 GB float64) + two δW files
# (5.6 GB bf16) + load transients (~25 GB peak expected); 48G as E44's stage-R rehearsal.
#SBATCH --job-name=commute-qwen_maps_diag
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
DW=$HOME/commute-data/results-large/qwen_updates/dW/gold_s0
set +e
.venv/bin/python experiments/qwen_maps/diag_stored.py --decomp "$COMMUTE_QWEN_DIR" \
  --e40-large "$HOME/commute-data/results-large/qwen_sft/nowarmup" \
  --dw "step2=$DW/step_2.safetensors" --dw "step100=$DW/step_100.safetensors" \
  --out experiments/qwen_maps/results/diag_stored.jsonl "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
