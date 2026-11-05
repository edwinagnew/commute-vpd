#!/bin/bash
# E99 Amendment 1: R-net clause (ii) on cased rows only, as the plan words it (S0-ii-A1, read from stage0.json's stored
# numbers, no model), then stage 3: forced-part and forbidden-part lowercase, seeds 0 and 1.
# Plan: reports/plans/2026-11-05_plan_e99-refine-verify-definition.md. Rules: experiments/definition_verify/results/
# predictions_amend1.json and predictions.json (amend1.py refuses unless both are committed unchanged).
#   cd ~/commute-agents/<role> && ~/commute-agents/fleet/link-data <role> && sbatch cluster/definition_verify_amend1.sbatch
# Exit 1 if a forced net share is off 1; exit 0 with nothing trained if the amended clause fails.
# Inputs: stage0.json (status pass); performed_anchor dW_lowercase_s{0,1}_step300.pt; induction_cue windows.pt; Pile val
# rows 0-255 and train rows 0-4095 (cached).
# Writes experiments/definition_verify/results/stage3_amend1.json, appends run.out; results/large/dW_{forced,forbidden}_
# lowercase_s{0,1}.pt (bf16, 57 MB each), reads_*.pt. Deletes nothing, commits nothing.
# Expected time on a debug L40 (definition_verify.sbatch's estimate; job 52787's stages 0-2 took 66 s and 16 s for the
# copy arms): start-up and Pile tokenizing ~2 min; stage 3 ~10 min if the four arms match near step 300, ~30 min if all
# hit the 3,000-step cap (arm step about 0.1 s, unmeasured). The limit is 1 h.
#SBATCH --job-name=commute-e99-amend1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS E99 amend1" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/definition_verify/amend1.py 2>&1 | tee -a experiments/definition_verify/results/run.out || rc=$?
echo "=== FINISHED E99 Amendment 1 (exit $rc) — results in $COMMUTE_REPO/experiments/definition_verify/results/ ==="
exit $rc
