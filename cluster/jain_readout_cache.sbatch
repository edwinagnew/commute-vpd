#!/bin/bash
# E59, step 1 of 2 (amended 2026-11-02 after job 48279's stop): cache the readouts' training rows, Pile train rows
# 32768-65535 (data.pile_rows("train", 32768, 32768), streamed from the HF dataset), before the GPU job.
#   cd ~/commute-agents/<role> && sbatch cluster/jain_readout_cache.sbatch   # then: sbatch cluster/jain_readout.sbatch
# CPU only. Writes artifacts/sets/train_32768_32768.npy (~134 MB; artifacts/ is a link into ~/commute-data, not git).
# Does nothing if the file exists. Commits nothing. Expected ~5 min (Fable's estimate; streaming skips 32768 rows first).
#SBATCH --job-name=commute-jain-readout-cache
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
.venv/bin/python -c 'from commute.data import pile_rows; r = pile_rows("train", 32768, 32768); print(tuple(r.shape), int(r[0, 0]), flush=True)'
