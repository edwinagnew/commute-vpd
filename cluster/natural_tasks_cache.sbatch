#!/bin/bash
# E124, before stage 0: cache Pile train rows START .. START+N-1 (data.pile_rows("train", START, N), streamed from the
# HF dataset). Plan: reports/plans/2026-11-06_plan_e124-natural-definition-tasks.md, "Revision before any run" item 4.
# Modelled on cluster/jain_readout_cache.sbatch. Two files are used:
#   cd ~/commute-agents/<role> && sbatch cluster/natural_tasks_cache.sbatch 65536 524288    # task rows (job 54712, done)
#   cd ~/commute-agents/<role> && sbatch cluster/natural_tasks_cache.sbatch 589824 524288   # more held-out windows
# CPU only. Writes artifacts/sets/train_<START>_<N>.npy (~2.1 GB int64 for N = 524288; artifacts/ is a link into
# ~/commute-data, not git) and prints its shape, first token, seconds and sha256. Does nothing if the file exists.
# Expected time: job 54712 (65536, 524288) took 210 s; the second file re-skips 589,824 rows first: ~5-8 min.
# Job 48321 hung at interpreter exit until cancelled, so this one leaves with os._exit after its print.
# Memory: np.stack of 524,288 rows of 512 int64 = 2.1 GB, ~4.5 GB peak with the row list; 24G as Fable's advice.
#SBATCH --job-name=commute-e124-cache
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS args=$*" >> cluster/logs/jobs.log
.venv/bin/python -c '
import hashlib, os, sys, time
from commute.data import SETS_DIR, pile_rows
start, n = int(sys.argv[1]), int(sys.argv[2])
t0 = time.time()
r = pile_rows("train", start, n)
p = SETS_DIR / f"train_{start}_{n}.npy"
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
print(p, tuple(r.shape), int(r[0, 0]), f"{time.time() - t0:.0f} s", "sha256", h.hexdigest(), flush=True)
os._exit(0)
' "$1" "$2"
echo "=== FINISHED E124 cache $1 $2 (exit 0) ==="
