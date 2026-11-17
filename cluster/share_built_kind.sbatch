#!/bin/bash
# E200 (cell P2.33): the 67M re-gating share on E139's fine-tunes, grouped by the kind of change built in, and Cue re-fine-tuned
# on 3 seeds. Plan: reports/plans/2026-11-17_plan_share-built-kind.md. Rules: experiments/share_built_kind/results/predictions.json
# (the run refuses unless it is committed unchanged). One L40, debug QoS, $0. Submit from the clone root:
#   sbatch --comment='E200 share built kind' cluster/share_built_kind.sbatch
# Stage 0 (identity control, alone) stops the job with exit 1 on a miss; nothing else runs. A rerun resumes after the last state read.
# Inputs: experiments/reuse_variable_test/results/{stage0.json,large/pieces.pt}; experiments/reuse_variable_finetune/results/
# large/{always_on,table,xor}_s{0,1,2}.pt, large/amend2_always_on_s{0,1,2}_step3000.pt, cue_s{0,1,2}.json; Pile rows; the 67M.
# Writes: results/stage{0,1,2}.json, results/cue_s<seed>_train.json, results/rules.json, results/large/cue_s<seed>.pt (268 MB each).
# Time on an L40: about 20 min (rows ~5, 12 reads ~4, three 300-step fine-tunes ~6, reads ~1); memory as E139's training, under 20 GB of the card.
#SBATCH --job-name=commute-e200
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/share_built_kind/run.py
