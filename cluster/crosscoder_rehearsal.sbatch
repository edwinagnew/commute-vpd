#!/bin/bash
# P2.31 run 2 (E203): the $0 rehearsal of every stage at tiny sizes (settings.rehearsal in
# experiments/crosscoder/results/run2/predictions.json: 1,536 latents, 4.1M tokens, 2,000 steps; permissive bars except
# the self control's exact ones). Gate 3 of reports/plans/2026-11-17_plan_p2-31-run2.md. Covers: the pinned
# dictionary-learning trainer on our stream (Pile rows from 32768, fresh key-value windows from the pool at 131072),
# the resume test (the self control run twice uninterrupted for the floor, then stopped at step 1000 with exit 3 and
# resumed; `resumecheck` compares step-matched losses and fails the job on a miss), the planted control, the timing
# stage, the four stage-1 keys one at a time (as on four pods), the CPU `1check`, and one stage-2 shard with its
# per-side bars. Outputs under experiments/crosscoder/results/run2_rehearsal/ and results/large/run2_rehearsal/
# (cleared first). Each stage logs its wall time; the job ends with "FINISHED <files>".
#   cd ~/commute-agents/compare-lead && ~/commute-agents/fleet/link-data compare-lead && sbatch --comment=E203 cluster/crosscoder_rehearsal.sbatch
# Inputs on NFS (gitignored): artifacts/vpd/, artifacts/sets/{train_32768_8192,train_65536_524288,val_0_64}.npy and the
# small val files the e53 loader reads, the states' results/large/ named in experiments/share_untrained/results/inventory.json
# and refit_96key's, and their task windows.
# Time: the old-code rehearsal (job 61005) took ~7 min; this one trains 3 more small crosscoders, ~12 min on an L40.
#SBATCH --job-name=commute-crosscoder-rehearsal
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
test -L experiments/refit_96key/results/large || { echo "results/large is not linked: run ~/commute-agents/fleet/link-data compare-lead"; exit 1; }
R=experiments/crosscoder/results/run2_rehearsal
rm -rf "$R" experiments/crosscoder/results/large/run2_rehearsal
T_JOB=$(date +%s)

step() {  # step NAME EXPECTED_EXIT ARGS...: run one stage, log its wall time, fail on an unexpected exit code
    local name=$1 want=$2 t0 rc=0
    shift 2
    t0=$(date +%s)
    echo "== $name: run.py $*"
    .venv/bin/python experiments/crosscoder/run.py "$@" --rehearse || rc=$?
    echo "== $name: exit $rc in $(( $(date +%s) - t0 )) s"
    [ "$rc" -eq "$want" ] || { echo "REHEARSAL FAIL at $name: exit $rc, expected $want"; exit 1; }
}

step "self A (uninterrupted)" 0 0a
cp "$R/stage0a.json" "$R/stage0a_A.json"
step "self B (uninterrupted, the floor)" 0 0a
cp "$R/stage0a.json" "$R/stage0a_B.json"
step "self C (stopped after step 1000)" 3 0a --stop-after 1000
ls -la experiments/crosscoder/results/large/run2_rehearsal/ckpt/
step "self C (resumed)" 0 0a
step "resume check" 0 resumecheck
step "planted" 0 0b
step "timing" 0 t
for key in reuse new random reuse_seed1; do step "stage 1 $key" 0 1 "$key"; done
step "stage 1 check" 0 1check
step "stage 2 shard 0/40" 0 2 0/40
echo "FINISHED rehearsal in $(( $(date +%s) - T_JOB )) s: $(ls "$R"/*.json "$R"/sides/*.json | tr '\n' ' ')"
