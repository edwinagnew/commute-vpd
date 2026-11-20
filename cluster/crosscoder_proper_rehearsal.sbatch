#!/bin/bash
# E235 (P2.49 run 1): the $0 rehearsal of every job at tiny dictionary size (settings.cc.rehearsal in
# experiments/crosscoder_proper/results/predictions.json: 3,072 latents, 4.1M tokens, 2,000 steps; full read and
# latent-scaling sizes; the exact self bars, the rest relaxed). Gate 3 of reports/plans/2026-11-20_plan_p2-49-crosscoder-proper.md.
#   part a: the resume test (self.A.c0 twice uninterrupted for the floor, then stopped after step 1000 with exit 3 and
#           resumed; score.py resumecheck), stage 0 as one pod, check0, the timing stage alone and two at once at the
#           full dictionary size (timingcheck), the scorecheck at full dictionary and read sizes, the e53 item (Pile fill)
#   part b: stage 1 as two pods, check1, and the hand-back path: tar two items, sha256, unpack into results/rehearse_dl/
#           and compare
# Submit from the clone root, part b after part a:
#   sbatch --comment='E235 rehearse: crosscoder_proper a' cluster/crosscoder_proper_rehearsal.sbatch a
#   sbatch --comment='E235 rehearse: crosscoder_proper b' --dependency=afterok:<a> cluster/crosscoder_proper_rehearsal.sbatch b
#SBATCH --job-name=commute-e235-rehearsal
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/crosscoder_proper/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data"; exit 1; }
E=experiments/crosscoder_proper
R=$E/results
PY=.venv/bin/python
T_JOB=$(date +%s)

step() {  # step NAME EXPECTED_EXIT CMD...: run, log wall time, fail on an unexpected exit code
    local name=$1 want=$2 t0 rc=0
    shift 2
    t0=$(date +%s)
    echo "== $name: $*"
    "$@" || rc=$?
    echo "== $name: exit $rc in $(( $(date +%s) - t0 )) s"
    [ "$rc" -eq "$want" ] || { echo "REHEARSAL FAIL at $name: exit $rc, expected $want"; exit 1; }
}

if [ "$1" = a ]; then
    rm -rf $R/rehearse $R/rehearse_A $R/rehearse_B $R/rehearse_dl $R/large/rehearse $R/large/rehearse_A $R/large/rehearse_B
    step "self A uninterrupted" 0 $PY $E/run.py item self.A.c0 --rehearse --tag rehearse_A
    step "self A again (the floor)" 0 $PY $E/run.py item self.A.c0 --rehearse --tag rehearse_B
    step "self A stopped after step 1000" 3 $PY $E/run.py item self.A.c0 --rehearse --stop-after 1000
    ls -la $R/large/rehearse/ckpt/
    step "self A resumed" 0 $PY $E/run.py item self.A.c0 --rehearse
    step "resume check" 0 $PY $E/score.py resumecheck --rehearse
    step "stage 0 as one pod" 0 $PY $E/run.py pod 0 0 1 --rehearse
    step "check0" 0 $PY $E/score.py check0 --rehearse --tag rehearse
    step "timing A alone" 0 $PY $E/run.py timing A solo --rehearse
    step "timing B alone" 0 $PY $E/run.py timing B solo_B --rehearse
    $PY $E/run.py timing A p0 --rehearse & P0=$!
    $PY $E/run.py timing A p1 --rehearse & P1=$!
    wait $P0
    wait $P1
    step "timing check" 0 $PY $E/score.py timingcheck --rehearse --tag rehearse
    step "e53 item (Pile fill)" 0 $PY $E/run.py item e53__lowercase_s0.A.c0 --rehearse
    step "scorecheck at full sizes" 0 $PY $E/run.py scorecheck --rehearse
else
    step "stage 1 pod 0 of 2" 0 $PY $E/run.py pod 1 0 2 --rehearse
    step "stage 1 pod 1 of 2" 0 $PY $E/run.py pod 1 1 2 --rehearse
    step "check1" 0 $PY $E/score.py check1 --rehearse --tag rehearse
    mkdir -p $R/rehearse_dl
    for key in gauss.A.c0 head.B.c0; do
        tar cf $R/rehearse_dl/$key.tar $R/rehearse/items/$key.json $R/large/rehearse/$key.pt
        sha256sum $R/rehearse_dl/$key.tar
        mkdir -p $R/rehearse_dl/$key && tar xf $R/rehearse_dl/$key.tar -C $R/rehearse_dl/$key
        for f in $R/rehearse/items/$key.json $R/large/rehearse/$key.pt; do
            a=$(sha256sum < "$f"); b=$(sha256sum < "$R/rehearse_dl/$key/$f")
            [ "$a" = "$b" ] && echo "download path ok: $f" || { echo "REHEARSAL FAIL: $f differs after tar"; exit 1; }
        done
    done
fi
echo "FINISHED rehearsal part $1 in $(( $(date +%s) - T_JOB )) s"
