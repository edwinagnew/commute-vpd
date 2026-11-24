#!/bin/bash
# P2.75 run 1: the $0 rehearsal of the diff-SAE (spec reports/plans/2026-11-23_plan_p2-75-diff-sae-spec.md, section 5).
# Full SAE size (1,536 latents, k 100), 12M training tokens (past the 10M-token AuxK threshold; a known dead set is
# written after step 5,000 and the AuxK loss must turn on), validation every 2,000 steps on 1M tokens; the stream, the
# normaliser (1M), latent scaling (50M, the pinned closed_form_scalars), the reading and the KL patches at full size;
# the saved SAE reloaded to identical encodings (no resume). First the $0 CPU fixtures (tests/test_diff_sae.py).
# results/predictions.json settings.sae.rehearsal and thresholds.rehearsal (every rehearsal-set bar null: recorded,
# not applied). Per item it measures ms per step, peak GPU memory, the pooled FVE per validation, the float32 against
# float64 floor of the numerator identity (scaler_checks), the KL floors (kl.*.kl_all.h_F, gap) and K.
# Each item is about 15-25 min on an L40 (stream and normaliser passes over 100M tokens), so two items a job:
#   sbatch --comment='E### rehearse: diff_sae' cluster/diff_sae_rehearse.sbatch plant.A.c0 head-r1.A.c0
#   sbatch --comment='E### rehearse: diff_sae' cluster/diff_sae_rehearse.sbatch head-r1.A.c1 random_d0-r1.A.c0
# Outputs: experiments/diff_sae/results/rehearse/items/<key>.json, results/large/rehearse/<key>.pt, and a tar + sha256
# per item in results/rehearse_dl/.
#SBATCH --job-name=commute-p275-rehearse
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/diff_sae/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data"; exit 1; }
[ $# -ge 1 ] || { echo "usage: sbatch ... cluster/diff_sae_rehearse.sbatch KEY [KEY ...]"; exit 2; }
E=experiments/diff_sae
R=$E/results
PY=.venv/bin/python
T_JOB=$(date +%s)
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

step() {  # step NAME CMD...: run, log wall time, fail on a non-zero exit
    local name=$1 t0 rc=0
    shift
    t0=$(date +%s)
    echo "== $name: $*"
    "$@" || rc=$?
    echo "== $name: exit $rc in $(( $(date +%s) - t0 )) s"
    [ "$rc" -eq 0 ] || { echo "REHEARSAL FAIL at $name: exit $rc"; exit 1; }
}

step "CPU fixtures" $PY -m pytest -q tests/test_diff_sae.py
for key in "$@"; do
    rm -f "$R/rehearse/items/$key.json"
    step "item $key" $PY $E/run.py item "$key" --rehearse
    mkdir -p $R/rehearse_dl
    tar cf "$R/rehearse_dl/$key.tar" "$R/rehearse/items/$key.json" "$R/large/rehearse/$key.pt"
    sha256sum "$R/rehearse_dl/$key.tar"
done
echo "FINISHED P2.75 rehearsal ($*) in $(( $(date +%s) - T_JOB )) s"
