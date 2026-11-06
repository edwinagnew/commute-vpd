#!/bin/bash
# E125, the repeat-label pair, one stage per job: 0 (W0: pool, windows, induction gap, possession probe and control, P0;
# exit 1 = "possession fails"), 1 (the four fine-tunes, delta-W in bf16 to results/large/, then E112's definition reads:
# random direction first, P1-P3 and verdicts, pruning and reverse beside), 2 (the decomposition side: route split of
# each fine-tune, E38 A2's removal of S, E34's student per state). Plan: reports/plans/2026-11-06_plan_e125-fresh-reuse-pair.md.
# Rules: experiments/fresh_reuse_pair/results/predictions.json (the run refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/fresh_reuse_pair.sbatch 0
# Then `... 1` once stage 0 exits 0, and `... 2` once stage 1 exits 0. Stages 1 and 2 may exit 3 (too little time left:
# resubmit the same stage, finished states are kept): ~/commute-agents/fleet/resubmit --code 3 cluster/fresh_reuse_pair.sbatch <N>
# Exit 0 done ("STAGE n DONE"); 1 a stop ("STAGE n STOPPED", reason under "stopped" in results/stage<N>.json).
# Inputs: Pile train rows 0-32767 and 32768-65535 (cached in artifacts/sets), the target model and decomposition in
# artifacts/vpd/, experiments/transport_test/results/large/alive.pt (stage 2), new_vs_existing/results/new_vs_existing_amend1.json.
# Writes results/stage<N>.json (after every step), results/stage<N>.out (this log, appended); results/large/task.pt (~10 MB),
# probes.pt, dW_{real,twin}_s{0,1}.pt (bf16, ~57 MB each), student_{real,twin}_s{0,1}.pt (bf16 offsets, ~60 MB each).
# Time on a debug L40 (estimates with sources, plan section "Time"): stage 0 ~3 min (E112's stage 0: 1 min after the model
# load, plus the repeat flags and 300-word probe positions on CPU); stage 1 ~8 min (E112: fine-tunes 0.06-0.09 s/step, the
# twin 300 steps; reads, pruning, reverse ~2.5 min); stage 2 ~17 min (E119: model, alive and least-squares context
# ~1.5 min, a student ~2-2.5 min; E38 A2's removal grid ~0.25 s per read at 100 windows). Memory: stage 2 keeps two
# W0 gate sets of 256 windows (~1.6 GB each, fp32) on the host; 32G as E119's.
#SBATCH --job-name=commute-e125
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
case "${1:-}" in
  0|1) cmd=(experiments/fresh_reuse_pair/run.py --stage "$1") ;;
  2) cmd=(experiments/fresh_reuse_pair/stage2.py) ;;
  *) echo "fresh_reuse_pair.sbatch: give the stage, 0, 1 or 2" >&2; exit 2 ;;
esac
rc=0
.venv/bin/python "${cmd[@]}" 2>&1 | tee -a "experiments/fresh_reuse_pair/results/stage$1.out" || rc=$?
echo "=== FINISHED E125 stage $1 (exit $rc) — results in $COMMUTE_REPO/experiments/fresh_reuse_pair/results/ ==="
exit $rc
