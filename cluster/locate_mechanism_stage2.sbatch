#!/bin/bash
# E96 stage 2: does a part of the 67M carry a fine-tune's gain on a task W0 already performs? E53's lowercase against its cipher,
# step 300, batch orders 0 and 1, no training. Plan: reports/plans/2026-11-05_plan_e96-locate-mechanism.md. Rules:
# experiments/locate_mechanism/results/predictions_stage2.json (stage2.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/locate_mechanism_stage2.sbatch
# Exit 1 at stage 0's miss or if the gradient check does not re-read stage 1. An outcome is not an exit code: read "outcome" in
# stage2_score.json.
# Inputs (read): experiments/performed_anchor/results/{finetune,read,a0}.json, large/dW_{lowercase,cipher}_s{0,1}_step300.pt (sha256
# against finetune.json); experiments/induction_cue/results/{run,predictions}.json, large/windows.pt, copy_s{0,2}.pt (sha256 against
# run.json); experiments/locate_mechanism/results/{means.pt,sweep.json,nominators.json} (stage 1's; means.pt's sha256 filed); the
# VPD artifacts, cached Pile rows. Never heads.pt.
# Writes experiments/locate_mechanism/results/stage2_{stage0,gradient,sweep,shares,beside,score}.json, stage2_mlp_scores.pt,
# stage2.out; refuses if stage2_stage0.json exists. Deletes nothing, commits nothing. Expected about 6 min on a debug L40: stage 1's
# 0.08 s per read of 200 x 41 tokens (run.out), scaled by tokens to 0.6 s per read of 500 x 127, ~450 reads, plus load.
#SBATCH --job-name=commute-e96-stage2
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
rc=0
.venv/bin/python experiments/locate_mechanism/stage2.py "$@" 2>&1 | tee -a experiments/locate_mechanism/results/stage2.out || rc=$?
echo "=== FINISHED E96 stage 2 (exit $rc) — results in $COMMUTE_REPO/experiments/locate_mechanism/results/ ==="
exit $rc
