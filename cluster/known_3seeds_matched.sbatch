#!/bin/bash
# E303 (cell P2.77): E187's size-matched share and pooled steering of E299's seed-2 states on their pairs' larger sides
# (e46/rand32_s2, e53/cipher_s2, e54/cipher_own_s2), so Fig 3a/4's matched series reads known3.json's states. Card
# ~/commute-data/cards/E303.md; code experiments/known_3seeds/matched.py (refuses unless E299's predictions.json, whose
# reader_share_tol it reads, is committed unchanged). Descriptive.
#   cd ~/commute-agents/baseline-lead && ~/commute-agents/fleet/link-data baseline-lead && sbatch --comment=E303 cluster/known_3seeds_matched.sbatch
# Control first, alone: e46/rand32_s0 and e53/cipher_s0 at their pair's c against m67.json (share r, pooled steering;
# tol 0.01); a miss prints "STOP: ..." and exits 1. Then "CONTROL DONE", the three states, "STAGE 3 DONE". A rerun resumes
# from results/stage3.json.
# Inputs (NFS, read only): the five updates (two s0 from share_untrained/results/inventory.json, three in
# known_3seeds/results/large/), the windows of e46/rand32, e53/cipher, e54/cipher, the saved JSONs, artifacts/vpd/.
# Writes experiments/known_3seeds/results/stage3.json (a few kB).
# Expected on a debug L40, from E299 job 66292 (stage 2, every reader: 47 s a key-value state, up to 3.3 min an e53
# state with its task loads): model load and SVDs about 2 min; e46 and e54 states about 1 min each, the e53 control
# about 3 min (its task's gates), e53/cipher_s2 after it about 1.5 min; total about 10 min, limit 30.
#SBATCH --job-name=commute-known-3seeds-matched
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --comment=E303
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root; caches in /ephemeral/$USER
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python experiments/known_3seeds/matched.py
