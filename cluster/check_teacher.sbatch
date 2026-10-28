#!/bin/bash
# E40 Amendment 1's teacher-training check on one cluster L40 (46 GB), debug QoS, $0: precision of the training
# forward and the positive control's fall on fixed rows (experiments/qwen_sft/check_teacher.py; diagnosis 2026-10-28).
# Reads the rehearsal's files (alive index, FineWeb pool, dW/step_3) from --out, read-only; writes
# experiments/qwen_sft/results/check_teacher.json and .log. Submit from the clone root; arguments go to the script:
#   sbatch cluster/check_teacher.sbatch --out /mnt/nw/home/e.agnew/commute-data/tmp/e40a1_rehearse
# Expected: ~12-22 min (325 fp32 steps of unmeasured length, bf16 ~0.9 s), GPU ~30 GB, host ~20 GB (U, V bf16 on the
# host while the offsets are built; the rehearsal's B peaked at 38 GB host with Γ too, which this job does not load).
#SBATCH --job-name=commute-check_teacher
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_sft/check_teacher.py --decomp "$HOME/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch" "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
