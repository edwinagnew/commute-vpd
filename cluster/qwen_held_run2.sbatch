#!/bin/bash
# P3.13 run 2's $0 rehearsal (E???, rung L3.2) on one cluster L40: experiments/qwen_held_readings/run2.py, the real code,
# limited to the controls and the pod's first state. Arguments pass to run2.py. Submit from the clone root, after
# results/run2/predictions.json is committed with its experiment number, with WANDB_API_KEY in the shell:
#   sbatch --comment='E??? rehearse: S controls + one library state' cluster/qwen_held_run2.sbatch \
#     --pair copy_cipher --code E??? --limit 1
# It reads: the do-nothing on cipher's rows (the row set's floor), the output-matrices-only change of cipher seed 0 at
# the late save (S >= 0.999), then that state at saved size, times E187's c (0.458), and its Gaussian change at both
# sizes: 6 S reads (2 controls, 4 of the state) at the full read sizes (128 rows of 256 tokens, 64 fit, 64 read), one 5.64 GB fetch.
# Measures per S read on an L40, the W&B fetch speed, peak GPU and host RAM, the library rows' do-nothing floor.
# Memory: run 1 peaked 50.8 GiB GPU on an H100 with every Gram on the GPU (results/run_pod.log); run2.py keeps the two
# fp64 Gram sets (17.5 GiB) on the host, so the GPU holds two fp32 models (15.2 GiB), G_fit^+ (8.75 GiB), one read's
# maps and cross moments (7 GiB) and logits (~5 GiB): ~36 GiB of 44.3. Host: Grams 18.8 GB + a state and its Gaussian
# 11.3 GB + a model's load 8 GB, ~40 GB (run 1's 99.8 GB peak was its share stage, which run 2 does not run).
# Records go to results/run2_rehearsal/<pair>/; the state lands in scratch on /mnt/nw and is deleted once read.
#SBATCH --job-name=commute-qwen_held_run2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub
nvidia-smi --query-gpu=name,memory.total --format=csv
S=$HOME/commute-data/scratch/qwen-lead/run2_rehearsal_$SLURM_JOB_ID
mkdir -p "$S"
trap 'rc=$?; rm -rf "$S"; echo "=== FINISHED (exit $rc) ==="' EXIT
.venv/bin/python experiments/qwen_held_readings/run2.py "$@" \
  --out experiments/qwen_held_readings/results/run2_rehearsal --fetch-to "$S"
