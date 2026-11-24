#!/bin/bash
# P2.67 (E284): the 10,000 Pile val document starts delta_bar is taken over (CPU; downloads monology/pile-uncopyrighted
# val.jsonl.zst). Usage, from the clone root:
#   sbatch --comment='E284 texts' cluster/adl_published_texts.sbatch
# Writes experiments/adl_published/results/texts.json (commit it) and results/large/doc_starts.pt.
#SBATCH --job-name=commute-adl-pub-texts
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log
.venv/bin/python experiments/adl_published/run.py texts
echo "=== FINISHED (exit $?) ==="
