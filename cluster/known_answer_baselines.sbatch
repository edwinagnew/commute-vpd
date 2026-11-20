#!/bin/bash
# E227 (cell P2.43): the re-gating share, effective rank and neurons for half the gain on E214's known-answer items
# (do-nothing, new_few, new_thin, x2, x10, moved, rebuilt). Plan reports/plans/2026-11-20_plan_p2-43-known-answer-baselines.md.
# Settings, expected values and rules: experiments/known_answer_baselines/results/predictions.json (run.py refuses unless
# it is committed unchanged). One job; stage 0 (the controls) runs first inside it and exits 1 on a miss:
#   cd <clone> && ~/commute-agents/fleet/link-data .
#   sbatch --comment=E227 cluster/known_answer_baselines.sbatch
# Writes experiments/known_answer_baselines/results/{stage0.json,items/<item>.json,score.json,e214_build/edits.json};
# prints "STAGE 0 DONE", "ITEM k of 6 DONE", "E227 DONE"; a filed stop prints "STOP: ..." and exits 1. A rerun skips
# what is written. Inputs: the model in artifacts/vpd/, cached Pile rows (artifacts/sets val_0_256), E62's windows and
# run.json, copy24 (experiments/open_pair/results/large/amend1/windows_copy_trained.pt), E195's rebuilt_mlp.pt, E53's
# lowercase_s0 and the E136/E98/E141 records.
# Times on an L40, each a measured item x its count: model load ~1 min; E214's edit build and the do-nothing ~2.5 min
# (E226 rehearsal job 61964, its stage 0); k_half ~17 s per 500-row read and ~1.5 s per 75-97 rows (E172, job 59606);
# the share ~2 s per 64 rows (E161, job 59106: 53 s for 15 reads of 64 rows with the load), so ~45 s per item (all rows
# plus two halves = two full reads) and ~35 s for the control: about 9-12 min in all. Peak GPU ~10 GB (E226 rehearsal
# 10.3 GB), host under 10 GB (E211 ctl 6.1 GB). 30-min limit.
#SBATCH --job-name=commute-known-baselines
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as E222: l40-worker's /ephemeral was full (2026-11-18)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -f experiments/known_answer_baselines/results/predictions.json || { echo "file the predictions first"; exit 1; }
.venv/bin/python experiments/known_answer_baselines/run.py "$@"
