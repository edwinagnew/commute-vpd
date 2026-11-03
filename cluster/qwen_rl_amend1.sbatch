#!/bin/bash
# E69 Amendment 1 (qwen_rl): one state's greedy accuracy, or one forced opening, on one cluster L40. Plan
# reports/plans/2026-11-03_plan_e69-amend1.md; order in experiments/qwen_rl/RUN_BRIEF_amend1.md. Submit from the clone root.
#   sbatch cluster/qwen_rl_amend1.sbatch --stage saved --arm W0
#   sbatch cluster/qwen_rl_amend1.sbatch --stage saved --arm N --step 50 --seed 70
#   sbatch cluster/qwen_rl_amend1.sbatch --stage open --arm W0 --opening "We are"
# Exit 0 done; 1 a refusal; 2 a control missed ("stopped" in results/<run>/amend1/<label>.json).
# --mem: E69's accuracy peaked at 11.4 GB on the host (A0.json); a δW adds 5.6 while it loads.
#SBATCH --job-name=commute-qwen_rl_amend1
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
.venv/bin/python experiments/qwen_rl/amend1.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
