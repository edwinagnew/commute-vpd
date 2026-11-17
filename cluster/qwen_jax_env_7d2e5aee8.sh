#!/bin/bash
# (Re)build the JAX CUDA environment of param-decomp 7d2e5aee8 (tag public/2026-11-02), the release that opens
# Goodfire's second Qwen3-1.7B decomposition p-9707d6ff. Used only by experiments/qwen_new_decomp/{convert,
# jax_reference}.py. The read-only clone ~/commute-data/param-decomp-7d2e5aee8 is never built in (an editable
# install writes into its tree): this script makes its own local clone beside it and syncs there. CPU-only work, so
# run it from the clone root as a CPU job:
#   ~/commute-agents/fleet/cpu-run -c 4 --mem 16G -t 60 bash cluster/qwen_jax_env_7d2e5aee8.sh
# Extra `cuda` (CUDA 12.8 wheels), not the bundle README's `cuda13`: the cluster's L40s run driver 535.247.01
# (cluster/logs/commute-qwen_jax_reference-38812.out), and the repo README gives cuda13 for r580+ only, cuda for
# r525-r579 (param-decomp README lines 30-31). The venv lands at $QWEN_JAX7_DIR/param_decomp/.venv; call its python
# directly (`uv run` would re-sync without CUDA).
set -euo pipefail
export QWEN_JAX7_DIR=${QWEN_JAX7_DIR:-$HOME/commute-data/param-decomp-7d2e5aee8-env}
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$HOME/commute-data/uv-cache-qwen-jax7   # the worker cannot see the dev node's /ephemeral
export UV_LINK_MODE=copy

if [[ ! -d $QWEN_JAX7_DIR ]]; then
    git clone --quiet "$HOME/commute-data/param-decomp-7d2e5aee8" "$QWEN_JAX7_DIR"
fi
git -C "$QWEN_JAX7_DIR" checkout --quiet 7d2e5aee8
test "$(git -C "$QWEN_JAX7_DIR" rev-parse --short=9 HEAD)" = 7d2e5aee8
cd "$QWEN_JAX7_DIR"
uv sync --project param_decomp --frozen --no-dev --extra cuda   # CUDA 12.8 wheels (pre-r580 driver arm)
param_decomp/.venv/bin/python -c "import jax, orbax.checkpoint, tensorstore, safetensors, yaml; print('jax', jax.__version__)"
rm -rf "$UV_CACHE_DIR"
