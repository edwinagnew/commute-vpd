#!/bin/bash
# E59's P3 rule on the original model's own ranks (review reports/reviews/2026-11-02_review_e59-analysis.md, M1): W0's
# logits and the linearised head through readout.key_medians and readout.p3 on E54's 97 held-out windows, elicit and
# cipher, and the possession count at the model's output. A diagnostic: no training, no new predictions.
#   cd ~/commute-agents/<role> && sbatch cluster/jain_readout_m1.sbatch
# Inputs on NFS: experiments/keydependent_elicit/results/large/windows_{elicit,cipher}.pt; the target model in
# artifacts/vpd/; E54's stage1.json and E59's predictions.json and run.json (committed).
# Writes experiments/jain_readout/results/m1_check.json (small) and prints every number. Commits nothing.
# Expected: a few minutes, nearly all of it the model load. Expected values: m1_check.py's docstring.
#SBATCH --job-name=commute-jain-readout-m1
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/jain_readout/m1_check.py "$@"
