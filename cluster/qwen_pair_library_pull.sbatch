#!/bin/bash
# P3.5 run 2: experiments/qwen_pair_library/pull.py, every p35 W&B artifact (the library's saved states) into
# results/large/library/ with sha256 sums against the pods' records. CPU only, no GPU. A batch job, not fleet/cpu-run
# (a long transfer must outlive the session). Up to 48 files of 5.64 GB (~270 GB); at W&B's 30-45 MB/s about 2 h, plus
# one sha256 pass per file; files already there are kept, so a rerun resumes. Submit from the clone root, with
# WANDB_API_KEY in the submitting shell:
#   sbatch --comment='E### pull library states' cluster/qwen_pair_library_pull.sbatch
# Exit 1: a sha256 differs from the pod's record (results/library/pull.json .mismatched).
#SBATCH --job-name=commute-qwen_pair_library_pull
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
test -n "${WANDB_API_KEY:-}" || { echo "WANDB_API_KEY is not set"; exit 1; }
.venv/bin/python experiments/qwen_pair_library/pull.py
