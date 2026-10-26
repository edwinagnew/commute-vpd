#!/bin/bash
# Jupyter server on one L40, using the project venv. Submit with `cluster/jupyter.sh start` from the laptop
# (or `sbatch cluster/jupyter.sbatch` on the controller). It writes node/port/token to
# $COMMUTE_REPO/.jupyter-session (mode 600, gitignored) once the server answers, and removes it on exit.
# The node has no Slurm client tools; jupyter.sh resolves the node address and end time on the controller.
# Default: debug QoS (higher priority, 1 GPU, hard 2 h wall). For longer sessions:
#   sbatch --qos=normal --time=08:00:00 cluster/jupyter.sbatch
# The kernel dies when the walltime ends; save checkpoints/results under /mnt/nw as you go.
#SBATCH --job-name=commute-jupyter
#SBATCH --partition=compute
#SBATCH --account=c11
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out

set -euo pipefail
source "$SLURM_SUBMIT_DIR/cluster/env.sh"   # submit from the clone root
cd "$COMMUTE_REPO"
echo "$(date -Is) $SLURM_JOB_ID $SLURM_JOB_NAME qos=$SLURM_JOB_QOS" >> cluster/logs/jobs.log

SESSION=$COMMUTE_REPO/.jupyter-session
read -r PORT TOKEN < <(.venv/bin/python -c '
import secrets, socket
s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1], secrets.token_hex(24))')

cleanup() {
    [[ -n "${JPID:-}" ]] && kill "$JPID" 2>/dev/null || true
    # only remove the session file if it is still ours (a newer job may have replaced it)
    grep -qx "JOB_ID=$SLURM_JOB_ID" "$SESSION" 2>/dev/null && rm -f "$SESSION"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

echo "job $SLURM_JOB_ID on $SLURMD_NODENAME port $PORT; GPU $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name --format=csv,noheader)"
.venv/bin/jupyter server --no-browser --ip=127.0.0.1 --port="$PORT" --port-retries=0 \
    --IdentityProvider.token="$TOKEN" --ServerApp.root_dir="$COMMUTE_REPO" \
    --ServerApp.allow_remote_access=True &
JPID=$!

for _ in $(seq 120); do
    if .venv/bin/python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/api', timeout=2)" 2>/dev/null; then
        umask 077
        printf 'JOB_ID=%s\nNODE=%s\nPORT=%s\nTOKEN=%s\n' \
            "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$PORT" "$TOKEN" > "$SESSION.tmp"
        mv "$SESSION.tmp" "$SESSION"
        echo "server ready; session file written"
        break
    fi
    kill -0 "$JPID" 2>/dev/null || { echo "jupyter server exited early"; exit 1; }
    sleep 1
done
wait "$JPID"
