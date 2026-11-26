#!/bin/bash
# E313 (P2.83 run 3) (cell P2.83): the $0 rehearsal of the --long path (800M tokens; outputs under results/long/) at
# settings.sizes.rehearsal_long (1,536 latents, 8.2M tokens, 4 epochs; every read at full size): the self control, the
# two planted controls run 3 repeats, and stage-2 shards 16/26 (e53/cipher_s1, e62/copy_s0) and 22/26
# (e54/cipher_own_s1, a Gaussian update). Outputs under results/rehearsal/long/ and results/large/rehearsal/long/
# (cleared first). Plan reports/plans/2026-11-25_plan_p2-83-crosscoder-fix.md "Run 3".
#   cd <clone root> && ~/commute-agents/fleet/link-data . && sbatch --comment='E313 rehearse: long path, full read sizes, tiny dict' cluster/crosscoder_fig4_rehearse_long.sbatch
# The run line needs PRED_SHA in the environment when the clone's HEAD lacks the commit (run.py require_filed).
# Inputs on NFS (gitignored): artifacts/vpd/, artifacts/sets/{train_32768_8192,train_131072_65536,val_0_8192}.npy and
# the val files the e53 loader reads, the states' files named in the inventories, their task windows.
#SBATCH --job-name=commute-crosscoder-fig4-rehearse-long
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:50:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/crosscoder_fig4/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data ."; exit 1; }
R=experiments/crosscoder_fig4/results/rehearsal/long
rm -rf "$R" experiments/crosscoder_fig4/results/large/rehearsal/long
T_JOB=$(date +%s)
step() {  # step NAME EXPECTED_EXIT ARGS...: run one stage, log its wall time, fail on an unexpected exit code
    local name=$1 want=$2 t0 rc=0
    shift 2
    t0=$(date +%s)
    echo "== $name: run.py $*"
    .venv/bin/python experiments/crosscoder_fig4/run.py "$@" --rehearse || rc=$?
    echo "== $name: exit $rc in $(( $(date +%s) - t0 )) s"
    [ "$rc" -eq "$want" ] || { echo "REHEARSAL FAIL at $name: exit $rc, expected $want"; exit 1; }
}

step "self (long)" 0 0a --long
step "planted random seed 0 (long)" 0 0b random --seed 0 --long
step "planted answer_half seed 0 (long)" 0 0b answer_half --seed 0 --long
step "stage 2 shard 16/26 (long)" 0 2 16/26 --long
step "stage 2 shard 22/26 (long)" 0 2 22/26 --long
.venv/bin/python experiments/crosscoder_fig4/plot.py --results "$R" --out "$R/figure_rehearsal.png"
echo "FINISHED rehearsal in $(( $(date +%s) - T_JOB )) s: $(ls "$R"/*.json "$R"/sides/*.json | tr '\n' ' ')"
