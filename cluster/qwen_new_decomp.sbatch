#!/bin/bash
# E131: the torch port on both Qwen3-1.7B decompositions (p-e9fcffd5 "old", p-9707d6ff "new"). Plan:
# reports/plans/2026-11-07_plan_e131-new-decomposition.md; stages in experiments/qwen_new_decomp/run.py.
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_new_decomp.sbatch controls      # stages 0-1: controls (a)-(c), then the G / G-think rows
#   sbatch cluster/qwen_new_decomp.sbatch measure 2     # new G (then 3 new G-think, 4 old G, 5 old G-think, 6 new T, 7 old T)
# Needs: the converted new decomposition (qwen_new_decomp_convert.sbatch), the JAX reference
# (qwen_new_decomp_jax_reference.sbatch; controls only), results/predictions.json, HF Qwen/Qwen3-1.7B@70d244cc and
# openai/gsm8k in ~/.cache/huggingface/hub. Measure jobs refuse to start until controls.json passes and rows.json exists.
# Output: experiments/qwen_new_decomp/results/{controls,rows,measure_<decomp>_<rows>}.json, each after its stage.
# Expected on one L40 (timings from qwen_port torch_check job 38829: U, V fp32 to host 70 s, Γ bf16 26 s, the
# adversary 0.8 s per row pass at 512 tokens with g held, 21 passes per start):
#   controls ~20-30 min: (a) ~4 min, (b) ~5 min, (c) old load ~4 min + g and KL on 128 rows ~4 min, rows ~5 min.
#   measure, per stage: load ~5 min, g ~2 min, ci_kl ~2 min, adversary 22 passes x 128 rows x 0.8 s x (mean length
#   / 512) (+~15% for re-densifying g and the clean forward each pass): T and G-think (512 tokens) ~45 min,
#   G (~300 tokens) ~25 min; one stage per job is well inside 2 h.
# Memory: GPU peak ~37 GB (Γ phase: target fp32 6.9 + U, V bf16 9.8 + Γ bf16 17.1 + one row's g and readouts; the
# adversary phase: target 6.9 + U, V fp32 19.5 + ~8 per row); host ~35 GB (U, V fp32 19.5 + sparse g of 128 rows
# <= 6 + the HF load).
#SBATCH --job-name=commute-qwen_new_decomp
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/qwen_new_decomp/results/ ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot and openai/gsm8k are here already
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv/bin/python experiments/qwen_new_decomp/run.py "$@"
