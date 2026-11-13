#!/bin/bash
# P3.4 run 3, the cluster read of the null's matched states on one L40 (debug QoS, $0): experiments/qwen_rl_readable/
# read3.py --stage gpu (read2.py's reads, unchanged). Plan: reports/plans/2026-11-13_plan_qwen-rl-readable-run3.md.
# Submit from the clone root:
#   sbatch --comment=E178 cluster/qwen_rl_run3_read.sbatch --controls-only     # first and alone; may need one resubmit
#   sbatch --comment=E178 cluster/qwen_rl_run3_read.sbatch --states N_s0@<t>   # one state a job, as each lands
# States are read from results/run3_dl/<pod>/dW/step_<t>.safetensors, where the runner downloads them; a named state that
# is missing there exits 1. Exit 3: the time left was under 1.2 x the longest read; resubmit the same command (read
# states and controls are skipped). Run 2 measured ~16 min a state on an L40 (run2_reads/states/*.json "seconds" 918-960).
# Card: the fp32 model 6.8 GB, W0 + δW 5.6 GB, GSM8K at batch 128 x 512 ~15 GB of 48 GB; host 11.4 + 5.6 GB.
#SBATCH --job-name=commute-qwen_rl_run3_read
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_rl_readable/read3.py --stage gpu --wall-min 28 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
