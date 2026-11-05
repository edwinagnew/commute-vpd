#!/bin/bash
# E97 Amendment 1, job 2: stage 3. E97's own stage 3 first (run.stage3: rank-1, half-known and copy-24 against the 24-key
# cipher, ident32 against rand32, the constant-answer tasks beside only; its file stage3.json, its rule and a), then every
# state's gain-path edges (exact), its own circuit and cross-faithfulness on its key pair, and E97's located rule again
# with the edges' nominees added. Plan: reports/plans/2026-11-05_plan_e97-amend1.md. Rules: results/predictions_amend1.json.
#   cd ~/commute-agents/<role> && sbatch cluster/e97_amend1_stage3.sbatch    (after cluster/e97_amend1.sbatch ended with exit 0)
# amend1.py refuses unless the amendment's stage 0 passed; stage 3 exits 0 without running unless amend1_stage1.json "go"
# is true (the plan's tree). Resumes from stage3.json and amend1_stage3.json. Prints "STAGE 3 DONE", then "STAGE table DONE".
# Inputs: as cluster/e97_locator_stage3.sbatch. Writes experiments/locator_table/results/stage3.json, amend1_stage3.json,
# table_amend1.json, amend1.out. Deletes nothing, commits nothing.
# Expected about 20 min on a debug L40 (estimates, not measured): start-up 2 min; E97's stage 3 about 8 min (its sbatch's
# estimate: 25 states x ~190 reads of 73-106 windows of 41 tokens at ~0.05 s); edges about 8 min (25 states x 955 exact
# patches on <= 106 windows cut to 21 tokens, stacked ~7 per forward, ~10 s each; 6 W0 circuits and 18 state circuits on
# key pairs, ~5 s each); the table seconds.
#SBATCH --job-name=commute-e97-amend1-s3
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS stages=3" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/locator_table/amend1.py --stages 3 2>&1 | tee -a experiments/locator_table/results/amend1.out || rc=$?
echo "=== FINISHED E97 Amendment 1 stage 3 (exit $rc) — results in $COMMUTE_REPO/experiments/locator_table/results/ ==="
exit $rc
