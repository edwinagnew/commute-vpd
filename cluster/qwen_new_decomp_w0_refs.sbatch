#!/bin/bash
# E131 Amendment 1: the W0 references on the new decomposition (experiments/qwen_new_decomp/w0_refs.py; plan
# reports/plans/2026-11-09_plan_new-decomposition-reruns.md, step 0(b)). Two submissions of this one script:
#   cd ~/commute-agents/<role>
#   sbatch cluster/qwen_new_decomp_w0_refs.sbatch 0      # the control, on the OLD decomposition, alone
#   sbatch cluster/qwen_new_decomp_w0_refs.sbatch 1      # NEW; refused unless stage 0's file says pass
# Needs: results/predictions_amend1.json committed; the old stored files under
# ~/commute-data/results-large/qwen_sft/nowarmup; HF Qwen/Qwen3-1.7B@70d244cc in ~/.cache/huggingface/hub. No tokenizer.
# Output: experiments/qwen_new_decomp/results/w0_refs_stage<n>.json and .../results/large/w0_refs/<decomp id>/.
# Expected per submission, on one L40 (46068 MiB seen by Slurm; E131's measure job 55542 is the sibling): E131 old T,
# experiments/qwen_new_decomp/results/measure_old_T.json seconds: g 40 s, masked KL 54 s, adversary 2311 s per 128 rows
# of 512 tokens. Here: 2 adversaries 77 min, g statistics + g rows + masked KL ~6 min, loads ~5 min cold: ~90 min.
# Memory: GPU peak ~37 GB (E131's two phases: g phase target fp32 6.9 + U, V bf16 9.8 + Γ bf16 17.1 + one row, then
# target + U, V fp32 19.5 + ~8 per row of the adversary) on a 46 GB card; the sibling's load peak was 42.5 GiB only
# with Γ and fp32 components together, which this job never holds. Host: E131's job read 52.5 GB with a host copy of the
# components (none here) and the file-backed shard pages; sparse g of two blocks <= 12 GB: 56G.
#SBATCH --job-name=commute-qwen_new_decomp_w0_refs
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) — results in $COMMUTE_REPO/experiments/qwen_new_decomp/results/ ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot is here already
nvidia-smi --query-gpu=name,memory.total --format=csv
case "${1:?stage 0 or 1}" in
  0) decomp=$COMMUTE_QWEN_DIR ;;
  1) decomp=$COMMUTE_QWEN_NEW_DIR ;;
  *) echo "stage must be 0 or 1: $1" >&2; exit 2 ;;
esac
.venv/bin/python experiments/qwen_new_decomp/w0_refs.py --stage "$1" --decomp "$decomp"
