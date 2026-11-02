#!/bin/bash
# E59, step 2 of 2: a Jain et al. Def 1 readout per block of the 67M model, trained on Pile text, and the causal check
# on its direction in the twelve saved E54/E56 fine-tunes. Plan: reports/plans/2026-10-30_plan_baseline-next2-draft.md.
# Rules: experiments/jain_readout/results/predictions.json (run.py refuses unless it is committed unchanged).
#   cd ~/commute-agents/<role> && sbatch cluster/jain_readout_cache.sbatch   # first, CPU, caches the training rows
#   sbatch cluster/jain_readout.sbatch                                      # then this; or, after a miss past training:
#   sbatch cluster/jain_readout.sbatch --readouts experiments/jain_readout/results/large/readouts.pt   (~5 min)
# Stage 0 runs first and the job exits 1 on any miss, before any ablation of a fine-tuned state is scored: sha256 of
# the windows and the twelve states, W0's rates and every state's gain reproduce E54/E56 within 1e-6; readout training
# on train rows 32768-65535 (amended after job 48279: fresh tokens), saved at once; it stops on a non-finite loss, on a
# missed early check (step 250: last block <= 5.0; step 2000: <= the model's batch loss + 0.5) or if step 20 projects
# over 18 min (always logged); the last block's held-out loss within 0.1 nats of the linearised head's; P3; the
# random-direction share (log-odds, net of W0) on every state within 0.05; W0's positive control at l*. Then A per state
# (log-odds share net of W0), the same removal at the last block on every state (reported), P2 and the decision tree.
# Every scored number is printed to this log.
# Inputs on NFS (gitignored): experiments/keydependent_elicit/results/large/{windows_elicit,windows_cipher,
# {elicit,cipher,cipher_own}_s{0,1}}.pt; experiments/keydependent_patch/results/large/{elicit,cipher}_s{2,3,4}.pt
# (~113 MB each); Pile rows artifacts/sets/{train_32768_32768,val_0_256}.npy (cached; the first by the cache job); the
# target model and decomposition in artifacts/vpd/; E54's stage1.json, stage2.json and E56's job2.json (committed).
# Writes experiments/jain_readout/results/run.json (small) and results/large/readouts.pt (~620 MB, the four readouts,
# fp32; results/large is a gitignored symlink into ~/commute-data/results-large/jain_readout). Commits nothing.
# Expected: ~22 min (model load ~1; stage 0 inputs ~1-2; training 15.6 at 0.233 s/step measured in job 48279; losses,
# scale and held-in ~1; possession, controls, ablations ~2), under the 30-min wall with the 18-min training guard.
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
