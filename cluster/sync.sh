#!/usr/bin/env bash
# Run on the LAPTOP from anywhere. Moves the tree between the laptop and cluster NFS.
#
#   cluster/sync.sh up         # code + results (incl. gitignored results/large/) -> cluster; never deletes or
#                              # overwrites a newer file there
#   cluster/sync.sh artifacts  # artifacts/ (VPD decomposition + target model, ~3.2 GB) -> cluster
#   cluster/sync.sh down       # experiments/*/results + figures, cluster/logs <- cluster; never deletes, never
#                              # overwrites a newer local file (notebooks stay local: VS Code edits them here)
#
# After `up` with a changed pyproject.toml/uv.lock, rebuild the venv on a GPU node: sbatch cluster/setup.sbatch.
set -euo pipefail

HOST=${COMMUTE_CLUSTER:-e.agnew@185.141.218.207}
REMOTE=/mnt/nw/home/e.agnew/commute
cd "$(dirname "$0")/.."

case "${1:-}" in
up)
    rsync -a --update --info=stats1 \
        --exclude=/.venv/ --exclude=/.git/ --exclude=/artifacts/ --exclude=/remote-kernels/ \
        --exclude=/.jupyter-session --exclude=/cluster/logs/ \
        --exclude='.env' --exclude='.env.*' --exclude='__pycache__/' --exclude='.ipynb_checkpoints/' \
        --exclude='.pytest_cache/' --exclude='.ruff_cache/' \
        ./ "$HOST:$REMOTE/"
    ;;
artifacts)
    rsync -a --partial --info=stats1 artifacts/ "$HOST:$REMOTE/artifacts/"
    ;;
down)
    rsync -a --update --info=stats1 --prune-empty-dirs \
        --include='/experiments/' --include='/experiments/*/' \
        --include='/experiments/*/results/***' --include='/experiments/*/figures/***' \
        --include='/cluster/' --include='/cluster/logs/***' \
        --exclude='.ipynb_checkpoints/' --exclude='*' \
        "$HOST:$REMOTE/" ./
    ;;
*)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
