#!/bin/bash
# E244 (P2.52) gate 3: $0 rehearsal at E203's rehearsal crosscoder sizes. neuron 0/15 (full read sizes); the control
# stage, xc controls 0/3, 1/3, 2/3 (the do-nothing, then the copy and prior_cipher repeats: the repeat path); 1/3 again
# (resume from its saved side); then a split of the other 28 items, shards 0/14 and 13/14 (two items each: scaled and
# random), and 0/14 again (resume). Each step must exit 0; the job exits 1 at the first that does not, and at the end
# checks the files the stages wrote. Plan: reports/plans/2026-11-21_plan_p2-52-key-grid-baselines.md. From the clone root:
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
X=$R/rehearsal/crosscoder
PY=.venv/bin/python
test -L $R/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data"; exit 1; }
export PRED_SHA=$(git show HEAD:$R/predictions.json | sha256sum | cut -d' ' -f1)
T_JOB=$(date +%s)

step() {  # step NAME LOG CMD...: run, tee the log, fail the job on a non-zero exit
    local name=$1 log=$2 t0 rc
    shift 2
    t0=$(date +%s)
    echo "== $name: $*"
    set +e
    "$@" 2>&1 | tee $log
    rc=${PIPESTATUS[0]}
    set -e
    echo "== $name: exit $rc in $(( $(date +%s) - t0 )) s"
    [ "$rc" -eq 0 ] || { echo "REHEARSAL FAIL at $name: exit $rc"; exit 1; }
}

rm -rf $R/rehearsal $R/large/rehearsal
step "neuron 0/15" $R/rehearsal_neuron_0-15.log $PY $E/run.py neuron 0/15 --rehearse
step "xc controls 0/3 (do-nothing)" $R/rehearsal_xc_c0.log $PY $E/run.py xc controls 0/3 --rehearse
step "xc controls 1/3 (copy repeat)" $R/rehearsal_xc_c1.log $PY $E/run.py xc controls 1/3 --rehearse
step "xc controls 2/3 (prior_cipher repeat)" $R/rehearsal_xc_c2.log $PY $E/run.py xc controls 2/3 --rehearse
step "xc controls 1/3 again (resume)" $R/rehearsal_xc_c1_b.log $PY $E/run.py xc controls 1/3 --rehearse
step "xc 0/14 (split)" $R/rehearsal_xc_0-14.log $PY $E/run.py xc 0/14 --rehearse
step "xc 13/14 (split)" $R/rehearsal_xc_13-14.log $PY $E/run.py xc 13/14 --rehearse
step "xc 0/14 again (resume)" $R/rehearsal_xc_0-14_b.log $PY $E/run.py xc 0/14 --rehearse

grep -q "0 crosscoders to train" $R/rehearsal_xc_c1_b.log || { echo "REHEARSAL FAIL: controls 1/3 resume retrained"; exit 1; }
grep -q "0 crosscoders to train" $R/rehearsal_xc_0-14_b.log || { echo "REHEARSAL FAIL: shard 0/14 resume retrained"; exit 1; }
jq() { $PY -c 'import json,sys; d=json.load(open(sys.argv[-1])); sys.exit(0 if d.get("pass") else 1)' "$@"; }  # no jq binary on the compute node
jq -e '.pass' $X/stage0a.json > /dev/null || { echo "REHEARSAL FAIL: stage0a did not pass"; exit 1; }
for f in $X/controls_0_3.json $X/controls_1_3.json $X/controls_2_3.json $X/shard_0_14.json $X/shard_13_14.json; do
    jq -e '.pass' $f > /dev/null || { echo "REHEARSAL FAIL: $f did not pass"; exit 1; }
done
n=$(ls $X/sides/*.json | wc -l)
[ "$n" -eq 6 ] || { echo "REHEARSAL FAIL: $n sides written, expected 6 (2 repeats + 2 x 2 shard items)"; exit 1; }
ls $X/sides
echo "REHEARSAL PASS: e244 rehearsal in $(( $(date +%s) - T_JOB )) s"
