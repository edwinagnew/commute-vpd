#!/bin/bash
# E57 (Q3) Stage C0 or C on one cluster L40, normal QoS ($0; Qwen-line exception: submit only when the queue is
# empty): experiments/qwen_reading/run.py --stage C0|C. Plan: reports/plans/2026-10-30_plan_q3-reading-the-update.md,
# sections 4, 6.2 and 7. Needs Stage A done; C refuses to start before C0 has finished. Submit from the clone root;
# not submitted by the code writer:
#   sbatch cluster/qwen_reading_C.sbatch C0     # the two random controls, alone, ~60 min; read C0.json, then
#   sbatch cluster/qwen_reading_C.sbatch C      # E40 step 10, gold, anchor 2 s0/s1: ~2.5 h, resumes per state
# Early stops (exit 1, results/stopped.json): C0's two 16-row KLs against Amendment 2; C's E40 K00, K0 against
# Amendment 2 and each E45 state's own-row KL against its E record and its sha256 against Stage A's. Past 2 h 45 min
# the job exits 0 between states: resubmit the same command. Per state: split ~20 min (Amendment 2's S), cells
# ~13 min. GPU ~42.5 GiB in the cells; Γ is parked on the host during Σ and the split. Host: Σ 11 GB, Γ parked
# 17 GB, pinned weight sets and X ~25 GB: 96G (plan section 8).
#SBATCH --job-name=commute-qwen_reading_C
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
stage=${1:?usage: sbatch cluster/qwen_reading_C.sbatch C0|C}
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_reading/run.py --stage "$stage" --decomp "$COMMUTE_QWEN_DIR" "${@:2}"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
