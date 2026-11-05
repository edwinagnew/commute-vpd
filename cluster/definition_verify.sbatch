#!/bin/bash
# E99 stages 0-3: W0 and the native states re-read against E97 (stage 0, alone, stops on a miss), forced-part copy (stage 1,
# stops if a net share is off 1), forbidden-part copy (stage 2), forced and forbidden lowercase (stage 3, only if stage 0's
# clause (ii) passed). Plan: reports/plans/2026-11-05_plan_e99-refine-verify-definition.md.
# Rules: experiments/definition_verify/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/definition_verify.sbatch
# (link-data makes experiments/definition_verify/results/large, a symlink into ~/commute-data/results-large/.)
# Exit 1 at a stage-0 miss or a forced net share off 1. A later stage alone: `--export=ALL,STAGES=2,3` (needs stage0.json "pass").
# Inputs (sha256 against each experiment's record, via locator_table/sites.py): induction_cue copy/lookup_s{0,1}.pt and
# windows.pt; performed_anchor dW_{lowercase,cipher}_s{0,1}_step300.pt; Pile val rows 0-255 and train rows 0-4095 (cached).
# Writes experiments/definition_verify/results/stage{0,1,2,3}.json, run.out; results/large/dW_*.pt (bf16, 57 MB each, 8 files),
# reads_*.pt. Deletes nothing, commits nothing.
# Expected time on a debug L40, from E97's run.out (whole-row read about 1.2 s: 1282 s for stage 2's ~1,000 reads; a read of
# E62's windows about 0.07 s): start-up and Pile tokenizing ~2 min; stage 0 ~4 min (63 W0 whole-row reads, 4 whole-row
# natives x 22 reads); stages 1-2 ~3 min (<= 300 steps each, a check every step); stage 3 ~10 min if both arms match near
# step 300, ~30 min if all four hit the 3,000-step cap (arm B step about 0.1 s, unmeasured: E53 logged no step times).
# Worst case ~40 min; the limit leaves twice that.
#SBATCH --job-name=commute-e99-definition-verify
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS stages=${STAGES:-0,1,2,3}" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/definition_verify/run.py --stages "${STAGES:-0,1,2,3}" 2>&1 | tee -a experiments/definition_verify/results/run.out || rc=$?
echo "=== FINISHED E99 stages ${STAGES:-0,1,2,3} (exit $rc) — results in $COMMUTE_REPO/experiments/definition_verify/results/ ==="
exit $rc
