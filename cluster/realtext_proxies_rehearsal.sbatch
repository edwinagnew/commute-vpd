#!/bin/bash
# P2.53 (E246): the $0 rehearsal of experiments/realtext_proxies/run.py at full read sizes, seed 0 of each pair's first
# side: pod 0, pod 1, then each again (the resume must skip what is written). Submit from the clone root:
#   sbatch --comment='E246 rehearse: rerun after device fix on cluster' cluster/realtext_proxies_rehearsal.sbatch
#SBATCH --job-name=commute-p253-rehearsal
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
test -L experiments/realtext_proxies/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data"; exit 1; }
E=experiments/realtext_proxies
R=$E/results
PY=.venv/bin/python
export PRED_SHA=0224baae70f9405d8b68e5a2f76cf407a8cfce4f77e2bb07ee027dba45e6b513
T_JOB=$(date +%s)

step() {  # step NAME CMD...: run, log wall time, fail on a non-zero exit
    local name=$1 t0 rc=0
    shift
    t0=$(date +%s)
    echo "== $name: $*"
    "$@" || rc=$?
    echo "== $name: exit $rc in $(( $(date +%s) - t0 )) s"
    [ "$rc" -eq 0 ] || { echo "REHEARSAL FAIL at $name: exit $rc"; exit 1; }
}

rm -f $R/pod0.json $R/pod1.json
step "pod 0 rehearse" $PY $E/run.py pod 0 --rehearse
step "pod 1 rehearse" $PY $E/run.py pod 1 --rehearse
step "pod 0 again (resume)" $PY $E/run.py pod 0 --rehearse
step "pod 1 again (resume)" $PY $E/run.py pod 1 --rehearse
mkdir -p $R/rehearsal
mv $R/pod0.json $R/rehearsal/pod0.json
mv $R/pod1.json $R/rehearsal/pod1.json
echo "FINISHED P2.53 rehearsal in $(( $(date +%s) - T_JOB )) s; outputs in $R/rehearsal/"
