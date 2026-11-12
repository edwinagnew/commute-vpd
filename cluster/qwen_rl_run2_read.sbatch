#!/bin/bash
# P3.4 run 2, the cluster read of the saved states on one L40 (debug QoS, $0): experiments/qwen_rl_readable/read2.py
# --stage gpu. Plan: reports/plans/2026-11-12_plan_qwen-rl-readable-run2.md, "After the pods". Submit from the clone root:
#   sbatch --comment=E### cluster/qwen_rl_run2_read.sbatch --controls-only           # first, before or while the pods run
#   sbatch --comment=E### cluster/qwen_rl_run2_read.sbatch --states R_s0@150,N_s0@150  # as states land (run2.py --chosen)
#   sbatch --comment=E### cluster/qwen_rl_run2_read.sbatch                            # every kept state not read yet
# The controls job (W0, E69's R at step 100) runs first and alone; state jobs refuse until it is done. A state job reads
# its states in order, one file each (results/run2_reads/states/), so jobs with different --states run side by side; it exits 3 when the
# time left is under 1.2 x the longest state so far: resubmit the same command, read states are skipped. Time per state on
# an L40 is measured by the controls job (log "seconds"); E104's full-model GSM8K states took 9-20 min on an L40.
# Split the states into jobs of one or two (time per state is the controls job's "seconds"), several L40s at once.
# Card: the fp32 model 6.8 GB, W0 + δW 5.6 GB, GSM8K at batch 128 and window 512 ~15 GB of cache (E69's reads ran in
# 80 GB beside Adam; an L40 has 48 GB). Host: E69's accuracy peak 11.4 GB plus a δW of 5.6 GB as it loads.
#SBATCH --job-name=commute-qwen_rl_run2_read
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
.venv/bin/python experiments/qwen_rl_readable/read2.py --stage gpu --wall-min 28 "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
