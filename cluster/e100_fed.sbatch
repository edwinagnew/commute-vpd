#!/bin/bash
# E100 fed_or_written: each saved 67M fine-tune's credit per part split into own (the part's update reverted) and inherited
# (carry - own), with the graft beside; read fed / written / mixed. No training. Plan: reports/plans/2026-11-05_plan_e100-fed-or-written.md.
# Rules: experiments/fed_or_written/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/e100_fed.sbatch
# Stages: 0 the two plants (exit 1 on a miss); 1 the poles (E99's forced and forbidden copy if saved, E62 copy and lookup,
# E53 cipher on whole rows; writes the bars; exit 1 unless the copy bar is below the cipher bar); 2 every other saved state.
# Each stage prints "STAGE <n> DONE" once its file is complete. Exit 1 also when all 24 reverted reads G off 0 on a state.
# Resume: a stage file keeps its finished states and a resubmit reads only the rest; `--export=ALL,STAGES=2` for one stage.
# Inputs (read, sha256 against each experiment's record): E97's sites and states (experiments/locator_table/sites.py), and
# definition-lead's E99 dW_{forced,forbidden}_copy_s{0,1}.pt in ~/commute-data/results-large/definition_verify (read only).
# Writes experiments/fed_or_written/results/stage{0,1,2}.json, table.json, run.out. Deletes nothing, commits nothing.
# Expected about 14 min on a debug L40, from E97's timings (locator_table/results/stage1.json and stage2.json `seconds`:
# 15-19 s for ~190 reads per E62 state, 253-323 s per whole-row state; run.out: stage 0 25 s, stage 1 99 s). E100 makes 85-88
# reads per state (28 parts or 29 with a pair, three reads each, and the all-reverted read): stage 0 about 0.5 min; stage 1
# about 5.5 min (10 E62-sized states at ~8 s, 2 cipher whole-row states at ~130 s); stage 2 about 6.5 min (2 lowercase
# whole-row states at ~130 s, 24 key-table and trigger states at ~4-8 s); start-up and file checks about 1.5 min.
#SBATCH --job-name=commute-e100-fed
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS stages=${STAGES:-0,1,2}" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/fed_or_written/run.py --stages "${STAGES:-0,1,2}" 2>&1 | tee -a experiments/fed_or_written/results/run.out || rc=$?
echo "=== FINISHED E100 stages ${STAGES:-0,1,2} (exit $rc) — results in $COMMUTE_REPO/experiments/fed_or_written/results/ ==="
exit $rc
