#!/bin/bash
# P3.5: experiments/qwen_pair_library/library.py, one arm and seed per job, one L40, $0, for when the cluster can start
# an arm at submission (pods are the default). Run 2 (E164) ran from this file; since 2026-11-11 library.py is run 3
# (E165: built arms copy cipher known invented, seeds 0 1 2; peak 1e-5, warm-up 30, 300 steps; results/library_run3/).
# Arguments are passed through; the submitter adds --comment. From the clone root:
#   sbatch --comment='E165 library run 3: copy seed 0' cluster/qwen_pair_library_run2.sbatch --arm copy --seed 0
# No --upload here: states are written straight to results/large/library_run3/ (the symlink into
# ~/commute-data/results-large/qwen_pair_library), at most 3 of 5.64 GB an arm-seed.
# Memory: run 2 peaked 35.6 GiB GPU, 18.7 GB host on an H100 (results/logs_built), so 32G host. Time: run 2 measured
# ~9.6 s/step on an H100; an L40 ran E61's arms at 16-17 s/step, so 300 steps ~1.4 h plus 12 tracked reads and the read
# stage (~25 min on an L40): about 2 h; normal QoS, 4 h.
# Log lines to wait on: "STAGE control DONE", "STAGE train DONE", "STAGE read DONE". Exit 1: a stop
# (results/library_run3/stopped_<arm>_s<seed>.json) or a refusal.
#SBATCH --job-name=commute-qwen_pair_library_run2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE=$HOME/.cache/huggingface/hub   # the model snapshot is here already
nvidia-smi --query-gpu=name,memory.total --format=csv
set +e
.venv/bin/python experiments/qwen_pair_library/library.py "$@"
status=$?
echo "=== FINISHED (exit $status) ==="
exit $status
