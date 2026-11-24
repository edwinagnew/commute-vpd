#!/bin/bash
# P2.72 (E293) rehearsal, $0: every stage of run 1 at the rehearsal sizes of experiments/delta_crosscoder/results/
# predictions.json (the full 3,840-latent dictionary, batch 4,096 and generation batch 2,000; 1,500 steps, 4,000 prompts,
# one candidate with an 8-latent pool), on one L40. Run plan reports/plans/2026-11-23_plan_p2-72-run.md.
#   part a: the data files (rehearsal, then the full 370 MB pre file once), the six continuation sets, C0a C0b C2A C2B
#   part b: rung1 stopped after step 700 (exit 3) and resumed, the other four C1 fits, the full-size scoring slice on
#           rung5 (one candidate, its 60-latent pool, 256 Pile-loss rows), and the hand-back (tar, sha256, unpack, cmp)
# It measures: ms per step and seconds per chunk (items/<key>.json curve, chunk_seconds), the sampler's tokens per second
# at the full generation batch (gen/<key>.json first_shard), minutes per latent read at full read sizes
# (items/rung5.slice.json), peak memory per stage (peak_gb_train, peak_gb, first_shard.peak_gb), and the C0a bf16
# deviation (items/C0a.json read.max_rn_dev) that thresholds.full.c0a_rn_dev is filed from.
# Submit from the clone root, part b after part a (predictions.json committed first; require_filed checks HEAD):
#   sbatch --comment='E293 rehearse: delta_crosscoder a' cluster/delta_crosscoder_rehearse.sbatch a
#   sbatch --comment='E293 rehearse: delta_crosscoder b' --dependency=afterok:<a> cluster/delta_crosscoder_rehearse.sbatch b
#SBATCH --job-name=commute-dxc-rehearse
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/delta_crosscoder/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
E=experiments/delta_crosscoder
R=$E/results
PY=.venv/bin/python
W=/ephemeral/$USER/dxc_rehearse
mkdir -p "$W"

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
    rm -rf $R/rehearse $R/large/rehearse $R/rehearse_dl
    step "prep rehearsal" 0 $PY $E/run.py prep --rehearse
    [ -f $R/data_full.json ] || step "prep full (CPU, 370 MB)" 0 $PY $E/run.py prep
    step "continuations" 0 $PY $E/run.py gen M gauss.A gauss.B head-r1.A head-r5.A head.B --rehearse
    step "C0a, C0b, C2 at A and B" 0 $PY $E/run.py fit C0a C0b C2A C2B --rehearse
elif [ "$1" = b ]; then
    step "rung1 stopped after step 700" 3 $PY $E/run.py fit rung1 --rehearse --stop-after 700 --c2-yes-reported
    ls -la $R/large/rehearse/ckpt/
    step "C1 (rung1 resumes)" 0 $PY $E/run.py fit rung1 rung5 e212 headB rung5tied --rehearse --c2-yes-reported
    jq -e '.resumed_at == 700' $R/rehearse/items/rung1.json
    step "full-size scoring slice on rung5" 0 $PY $E/run.py slice rung5 --rehearse
    # the hand-back as a pod-runner does it: one archive per fit, sha256 at both ends, unpacked where it is read
    for k in C0a rung5; do
        tar czf "$W/$k.tgz" $R/rehearse/items/$k.json $R/large/rehearse/$k.pt $(ls $R/rehearse/items/$k.steer.json 2>/dev/null)
        ( cd "$W" && sha256sum "$k.tgz" > "$k.sha256" && sha256sum -c "$k.sha256" )
        mkdir -p $R/rehearse_dl/$k && tar xzf "$W/$k.tgz" -C $R/rehearse_dl/$k
        cmp $R/rehearse/items/$k.json $R/rehearse_dl/$k/$R/rehearse/items/$k.json
        cmp $R/large/rehearse/$k.pt $R/rehearse_dl/$k/$R/large/rehearse/$k.pt
    done
    du -sh $R/large/rehearse $W
fi
echo "== summary"
for f in $R/rehearse/gen/*.json; do jq -c '{key, first_shard, minutes}' "$f"; done
for f in $R/rehearse/items/*.json; do
    case "$f" in *.steer.json|*.slice_steer.json) continue ;; esac
    jq -c '{key, label: .read.label, minutes, peak_gb_train, peak_gb, chunk_seconds, ms_per_step: (.curve // [] | map(.ms_per_step) | if length > 1 then .[1:] | add / length else null end), step0, max_rn_dev: .read.max_rn_dev, minutes_per_latent, stopped}' "$f"
done
echo "=== FINISHED part $1 — outputs in $COMMUTE_REPO/$R/rehearse ==="
