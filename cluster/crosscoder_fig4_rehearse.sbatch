#!/bin/bash
# E311 (cell P2.83): the $0 rehearsal of every stage of experiments/crosscoder_fig4/run.py, full read sizes with a tiny
# dictionary (settings.sizes.rehearsal: 1,536 latents, 4.1M tokens, 2,000 steps; 8,192 val rows, 100k latent-scaling
# tokens, the full score windows), permissive bars except the self control's exact ones. Plan
# reports/plans/2026-11-25_plan_p2-83-crosscoder-fix.md. Covers: the self control three times (two uninterrupted for
# the floor, one stopped after step 500 with exit 3 and resumed) and `resumecheck`; the three planted variants (the
# answer one with --long); stage-2 shards 16/26 (e53/cipher_s1, e62/copy_s0) and 22/26 (e54/cipher_own_s1, the
# Gaussian update at e54/elicit_s0's norms). Outputs under experiments/crosscoder_fig4/results/rehearsal/ and
# results/large/rehearsal/ (cleared first). Each stage logs its wall time; the job ends with "FINISHED rehearsal".
#   cd <clone root> && ~/commute-agents/fleet/link-data . && sbatch --comment='E311 rehearse: full sizes, tiny dict' cluster/crosscoder_fig4_rehearse.sbatch
# The run line needs PRED_SHA in the environment when the clone's HEAD lacks the commit (run.py require_filed).
# Inputs on NFS (gitignored): artifacts/vpd/, artifacts/sets/{train_32768_8192,train_131072_65536,val_0_8192}.npy and
# the val files the e53 loader reads, the states' files named in the inventories, their task windows.
#SBATCH --job-name=commute-crosscoder-fig4-rehearse
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
R=experiments/crosscoder_fig4/results/rehearsal
rm -rf "$R" experiments/crosscoder_fig4/results/large/rehearsal
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

step "self A (uninterrupted)" 0 0a
cp "$R/stage0a.json" "$R/stage0a_A.json"
step "self B (uninterrupted, the floor)" 0 0a
cp "$R/stage0a.json" "$R/stage0a_B.json"
step "self C (stopped after step 500)" 3 0a --stop-after 500
ls -la experiments/crosscoder_fig4/results/large/rehearsal/ckpt/
step "self C (resumed)" 0 0a
step "resume check" 0 resumecheck
step "planted random seed 0" 0 0b random --seed 0
step "planted answer_half seed 0" 0 0b answer_half --seed 0
step "planted answer long seed 0" 0 0b answer --seed 0 --long
step "stage 2 shard 16/26" 0 2 16/26
step "stage 2 shard 22/26" 0 2 22/26
.venv/bin/python experiments/crosscoder_fig4/plot.py --results "$R" --out "$R/figure_rehearsal.png"
echo "FINISHED rehearsal in $(( $(date +%s) - T_JOB )) s: $(ls "$R"/*.json "$R"/sides/*.json | tr '\n' ' ')"
