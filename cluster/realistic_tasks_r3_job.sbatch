#!/bin/bash
# E212 (cell P2.38 run 3): the crosscoder of P2.31/E203 on run 2's real-text fine-tunes. Plan
# reports/plans/2026-11-18_plan_p2-38-run3.md; rules experiments/realistic_tasks/results/predictions_run3.json (run3.py
# refuses unless it is committed unchanged). Two uses, from the clone root after ~/commute-agents/fleet/link-data <role>:
#   sbatch --comment=E212 cluster/realistic_tasks_r3_job.sbatch prep        # the texts file every job reads (~15 min)
#   sbatch --comment=E212 cluster/realistic_tasks_r3_job.sbatch rehearsal   # every stage at tiny size, $0 (~45 min)
# prep writes experiments/realistic_tasks/results/large/run3/texts.pt (~25 MB) and results/run3_prep.json; it downloads
# AG News and SST-2 and streams Wikipedia fr/sw/tl/yo inside the job (run 2's loaders). The rehearsal runs prep first if
# texts.pt is missing, then: the planted control stopped after step 700 (exit 3) and resumed (its file must record
# resumed_at 700), the Gaussian, the six sides, and score3.py; outputs under results/run3_rehearsal/ and
# results/large/run3_rehearsal/ (cleared first). Any other argument list goes to run3.py as it is (one full-size job on
# an H100 is a pod's work, not this script's: 80 GB-class card, ~55 min).
# Inputs on NFS: artifacts/vpd/, artifacts/sets/{train_32768_8192,val_0_64}.npy (rehearsal) or
# {train_32768_98304,val_0_8192}.npy (full), run 2's six seed-0 updates under results/large/run2/, and for prep run 2's
# inputs (artifacts/sets train_0_4096, val_0_256; E136's inventory and the e53 read windows).
#SBATCH --job-name=commute-realistic-r3
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
export HF_HOME=$HOME/commute-data/cache/hf TMPDIR=$HOME/commute-data/cache/tmp; mkdir -p "$HF_HOME" "$TMPDIR"  # as run 2's job (l40-worker's /ephemeral was full)
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/realistic_tasks/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data <role>"; exit 1; }
PY=.venv/bin/python
RUN=experiments/realistic_tasks/run3.py

if [ "$1" != "rehearsal" ]; then
    $PY $RUN "$@"
    exit
fi

R=experiments/realistic_tasks/results/run3_rehearsal
rm -rf "$R" experiments/realistic_tasks/results/large/run3_rehearsal
T_JOB=$(date +%s)

step() {  # step NAME EXPECTED_EXIT ARGS...: run one job, log its wall time, fail on an unexpected exit code
    local name=$1 want=$2 t0 rc=0
    shift 2
    t0=$(date +%s)
    echo "== $name: run3.py $*"
    $PY $RUN "$@" || rc=$?
    echo "== $name: exit $rc in $(( $(date +%s) - t0 )) s"
    [ "$rc" -eq "$want" ] || { echo "REHEARSAL FAIL at $name: exit $rc, expected $want"; exit 1; }
}

test -f experiments/realistic_tasks/results/large/run3/texts.pt || step "prep" 0 prep
step "planted (stopped after step 700)" 3 planted --rehearsal --stop-after 700
ls -la experiments/realistic_tasks/results/large/run3_rehearsal/ckpt/
step "planted (resumed)" 0 planted --rehearsal
grep -q '"resumed_at": 700' "$R/run3_planted.json" || { echo "REHEARSAL FAIL: the planted run did not resume at step 700"; exit 1; }
step "gauss" 0 gauss --rehearsal
for s in L1 L3 T1 T2 T3 T4; do step "side $s" 0 "$s" --rehearsal; done
$PY experiments/realistic_tasks/score3.py --rehearsal
echo "FINISHED rehearsal in $(( $(date +%s) - T_JOB )) s: $(ls "$R"/*.json | tr '\n' ' ')"
