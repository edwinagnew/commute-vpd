#!/bin/bash
# P2.49 run 2: the $0 rehearsal of the new paths at tiny dictionary size (run 1's settings.cc.rehearsal: 3,072 latents,
# 4.1M tokens; full read sizes; run 1's rehearsal bars, the self bars exact). Gate 3 of
# reports/plans/2026-11-21_plan_p2-49-run2-size-ladder.md. Run 1's rehearsal (jobs 62445, 62446, 62452) covered the
# trainer, the streams, the scoring, the resume and the tar hand-back; this covers what run 2 adds: the head scaled by a
# factor (scale 0 = self, the bisection at the top rung), E147's states and a random draw scaled to the top rung (the
# largest factors), e58 elicit's windows, and the ladder and read scripts. Rung 3 is run 1's rehearsal items.
# Submit from the clone root, with the E-number claimed and predictions committed:
#   sbatch --comment='E243 rehearse: crosscoder_proper run2' cluster/crosscoder_proper_run2_rehearsal.sbatch
#SBATCH --job-name=commute-p249-run2-rehearsal
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

rm -rf $R/rehearse2 $R/large/rehearse2 $R/rehearse2_dl
step "sizes (forwards, full read sizes)" 0 $PY $E/run2.py sizes --rehearse
step "self through the scaled path (scale 0)" 0 $PY $E/run2.py item self.A.c0 --rehearse
step "check0" 0 $PY $E/score2.py check0 --rehearse
step "round 1 (seed 0, rungs 1, 2, 4, 5)" 0 $PY $E/run2.py item head-r1.A.c0 head-r2.A.c0 head-r4.A.c0 head-r5.A.c0 --rehearse
step "ladder after round 1" 0 $PY $E/score2.py ladder --rehearse
step "stage 2 at rung 5: a draw, e58 elicit, cipher" 0 $PY $E/run2.py item random_d0-r5.A.c0 e58__elicit_s0-r5.A.c0 e58__cipher_s0-r5.A.c0 --rehearse
step "read at rung 5" 0 $PY $E/score2.py read --rehearse --x 5
mkdir -p $R/rehearse2_dl
key=e58__elicit_s0-r5.A.c0
tar cf $R/rehearse2_dl/$key.tar $R/rehearse2/items/$key.json $R/large/rehearse2/$key.pt
sha256sum $R/rehearse2_dl/$key.tar
echo "FINISHED run-2 rehearsal in $(( $(date +%s) - T_JOB )) s"
