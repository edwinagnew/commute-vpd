#!/bin/bash
# E128 (qwen_post_cutoff): the full Wikidata fetch into artifacts/post_cutoff/ (CPU job, gitignored, untrusted). One query at a
# time, a second apart, generic User-Agent, no personal detail. Submit from the clone root. A window file already on disk
# is skipped, so a resubmit continues. Plan: reports/plans/2026-11-06_plan_e128-post-cutoff-facts.md. Then build.py.
#SBATCH --job-name=commute-e128_fetch
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
.venv/bin/python -I experiments/qwen_post_cutoff/fetch.py --out artifacts/post_cutoff/full "$@"
