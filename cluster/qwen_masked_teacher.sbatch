#!/bin/bash
# E76, teacher training with masks in its loss ($0) on one cluster L40, normal QoS (Qwen-line exception: submit only when the queue
# is empty): experiments/qwen_masked_teacher/run.py. Plan: reports/plans/2026-11-03_plan_e76-masked-teacher.md. Submit from the clone
# root; not submitted by the code writer:
#   sbatch cluster/qwen_masked_teacher.sbatch
# g per row (0.35 GB x 72 rows, ~25 GB) goes to /ephemeral/$USER/e76 on the node (--out); another node computes it again.
# Outputs: experiments/qwen_masked_teacher/results/ (stage0.json, C/A/B/L.json, score.json, run.log; stopped.json on a stop, exit 1; a
# rerun refuses while it exists); results/large/ (cut, stage0_norm, <arm>_offsets bf16, <arm>_ckpt.pt while an arm runs). Past
# --wall-min (4 h 40 min, at a reading) it checkpoints and exits 0: resubmit, it resumes.
# Expected: load + stage 0 ~15 min, C ~15, A ~35, B ~55, L ~55, ~3 h in all by the advice's step times (0.82 s clean at half the
# tokens, ~2.5 s with the masked term; not measured with a backward). GPU, GB decimal: stage 0 ~46 (E74's reader with Γ); arms ~33
# (target 6.9, components bf16 9.8, offsets 4.4, δW bf16 2.8, masks 1.4 per row, ~7 as the rehearsal's training). Host peak ~60:
# g 25, components fp32 19.5 while building the offsets, δW 2.8, logits 2.5, a checkpoint ~3.
#SBATCH --job-name=commute-qwen_masked_teacher
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /ephemeral
set +e
.venv/bin/python experiments/qwen_masked_teacher/run.py --decomp "$COMMUTE_QWEN_DIR" --out "/ephemeral/$USER/e76" --wall-min 280 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
