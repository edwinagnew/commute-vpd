#!/bin/bash
# E124, before stage 0: cache the pronoun task's rows, Pile train rows 65536-589823 (data.pile_rows("train", 65536,
# 524288), streamed from the HF dataset). Plan: reports/plans/2026-11-06_plan_e124-natural-definition-tasks.md,
# "Revision before any run" item 4. Modelled on cluster/jain_readout_cache.sbatch.
#   cd ~/commute-agents/<role> && sbatch cluster/natural_tasks_cache.sbatch
# CPU only. Writes artifacts/sets/train_65536_524288.npy (~2.1 GB int64; artifacts/ is a link into ~/commute-data, not
# git) and prints its shape, first token and sha256. Does nothing if the file exists. Commits nothing.
# Expected time: job 48321 streamed 65,536 rows (32,768 skipped, 32,768 kept) in 26 s (jobs.log 11:10:17 -> file
# 11:10:43), ~2,500 rows/s; 589,824 rows here is ~4-10 min. The 2 h limit is debug's cap, kept as margin: no split
# needed. Job 48321 then hung at interpreter exit until cancelled, so this one leaves with os._exit after its print.
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
import hashlib, os, time
from commute.data import SETS_DIR, pile_rows
t0 = time.time()
r = pile_rows("train", 65536, 524288)
p = SETS_DIR / "train_65536_524288.npy"
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
print(tuple(r.shape), int(r[0, 0]), f"{time.time() - t0:.0f} s", "sha256", h.hexdigest(), flush=True)
os._exit(0)
'
echo "=== FINISHED E124 cache (exit 0) ==="
