#!/bin/bash
# E244 (P2.52) gate 3: $0 rehearsal. neuron 0/15 (full read sizes), then xc 14/15 at E203's rehearsal crosscoder sizes
# (shard 14/15 also runs the do-nothing crosscoder), twice: the second run tests the resume from saved files.
# Plan: reports/plans/2026-11-21_plan_p2-52-key-grid-baselines.md. Submit from the clone root:
#   sbatch --comment='E244 rehearse: key grid baselines' cluster/key_grid_baselines_rehearsal.sbatch
#SBATCH --job-name=commute-e244-rehearsal
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
E=experiments/key_grid_baselines
R=$E/results
PY=.venv/bin/python
test -L $R/large || { echo "results/large is not linked"; exit 1; }
export PRED_SHA=$(git show HEAD:$R/predictions.json | sha256sum | cut -d' ' -f1)
T_JOB=$(date +%s)
mkdir -p $R/rehearsal $R/large/rehearsal

step() {  # step NAME LOG CMD...
    local name=$1 log=$2 t0 rc=0
    shift 2
    t0=$(date +%s)
    echo "== $name: $*"
    ( "$@"; echo "=== FINISHED (exit $?) — results in $PWD/$R/rehearsal ===" ) 2>&1 | tee $log || true
    rc=${PIPESTATUS[0]}
    echo "== $name: done in $(( $(date +%s) - t0 )) s"
}

rm -rf $R/rehearsal/* $R/large/rehearsal/*
step "neuron 0/15" $R/rehearsal_neuron_0-15.log $PY $E/run.py neuron 0/15 --rehearse
step "xc 14/15 first" $R/rehearsal_xc_14-15_a.log $PY $E/run.py xc 14/15 --rehearse
step "xc 14/15 repeat (resume)" $R/rehearsal_xc_14-15_b.log $PY $E/run.py xc 14/15 --rehearse
echo "FINISHED e244 rehearsal in $(( $(date +%s) - T_JOB )) s"
