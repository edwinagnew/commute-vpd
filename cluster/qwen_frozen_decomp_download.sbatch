#!/bin/bash
# E175 (P3.9): download Goodfire's frozen-neuron Qwen3-1.7B decomposition (run p-015e0710, a 31 GB tar; user task
# 2026-11-13), check its sha256 and extract it. The presigned link (expires ~2026-11-19) is read at run time from
# ~/commute-data/frozen_decomp.txt so it never enters the repo.
#   cd ~/commute-agents/<role> && sbatch --comment='E175 ...' cluster/qwen_frozen_decomp_download.sbatch
# Resumable: curl -C - continues a partial tar; resubmit after a time-out. Output: $DEST/{p-015e0710-share.tar, data/}.
#SBATCH --job-name=commute-qwen_frozen_decomp_dl
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
trap 'echo "=== FINISHED (exit $?) ==="' EXIT
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> "$COMMUTE_REPO/cluster/logs/jobs.log"

DEST=/mnt/nw/home/e.agnew/commute-data/artifacts/p-015e0710-share
SHA=4659ca1b1e8b91dd9c7fb866218c22e22a552e93eda9cdb1e4399ed91809fb81
TAR=p-015e0710-share.tar
mkdir -p "$DEST" && cd "$DEST"
url=$(grep -o 'https://[^ ]*p-015e0710-share.tar[^ ]*' /mnt/nw/home/e.agnew/commute-data/frozen_decomp.txt | head -1)
test -n "$url" || { echo "no link in frozen_decomp.txt"; exit 1; }

if [ ! -f "$TAR.ok" ]; then
  for attempt in 1 2 3 4 5 6; do
    echo "=== curl attempt $attempt $(date -Is), have $(stat -c %s "$TAR" 2>/dev/null || echo 0) bytes ==="
    if curl -sS -f -L -C - --retry 5 --retry-delay 10 -o "$TAR" "$url"; then break; fi
    sleep 15
  done
  echo "=== downloaded $(stat -c %s "$TAR") bytes $(date -Is); sha256 ==="
  echo "$SHA  $TAR" | sha256sum -c - && touch "$TAR.ok"
fi
test -f "$TAR.ok"
echo "=== extracting $(date -Is) ==="
mkdir -p data && tar -C data -xf "$TAR"
echo "=== extracted $(date -Is) ==="
du -sh data
find data -maxdepth 4 | head -100
