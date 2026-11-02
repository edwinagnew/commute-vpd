#!/bin/bash
# E59, a Jain et al. Def 1 readout per block of the 67M model, trained on Pile text, and the causal check on its
# direction in the twelve saved E54/E56 fine-tunes. Plan: reports/plans/2026-10-30_plan_baseline-next2-draft.md.
# Rules: experiments/jain_readout/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/jain_readout.sbatch
# Stage 0 runs first and the job exits 1 on any miss, before any ablation of a fine-tuned state is scored: sha256 of
# the windows and the twelve states, W0's rates and every state's gain reproduce E54/E56 within 1e-6; readout training
# (exits at step 20 if it projects over 15 min, or on a non-finite loss); the last block's held-out Pile loss within
# 0.5 nats of the model's; P3; the random-direction share (log-odds, net of W0) on every state within 0.05; W0's
# positive control at l*. Then A per state (log-odds share net of W0, amended 2026-11-02 pre-run), the same removal at
# the last block on every state (reported, the positive control that A can read high), P2 and the decision tree.
# Every scored number is printed to this log.
# Inputs on NFS (gitignored): experiments/keydependent_elicit/results/large/{windows_elicit,windows_cipher,
# {elicit,cipher,cipher_own}_s{0,1}}.pt; experiments/keydependent_patch/results/large/{elicit,cipher}_s{2,3,4}.pt
# (~113 MB each); Pile rows artifacts/sets/{train_32768_8192,val_0_256}.npy (cached); the target model and
# decomposition in artifacts/vpd/; E54's stage1.json, stage2.json and E56's job2.json (committed).
# Writes experiments/jain_readout/results/run.json (small) and results/large/readouts.pt (~620 MB, the four readouts,
# fp32; results/large is a gitignored symlink into ~/commute-data/results-large/jain_readout). Commits nothing.
# Expected: ~8-15 min (model load ~1 min; stage 0 inputs ~1-2 min; 4000 readout steps of 96 x 41 tokens into four
# 768 x 50277 maps, ~3-8 min on an L40 with TF32 off; ablations ~1 min).
#SBATCH --job-name=commute-jain-readout
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
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/jain_readout/run.py "$@"
