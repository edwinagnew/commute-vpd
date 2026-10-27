#!/bin/bash
# (Re)build the JAX CUDA environment of param-decomp 177d0cce8, the code that wrote the Qwen3-1.7B decomposition
# (p-e9fcffd5). Used only by experiments/qwen_port/{convert,jax_reference}.py. A separate clone in ~/commute-data;
# the read-only ~/param-decomp is never built in. CPU-only work, so run it from the clone root as a CPU job:
#   ~/commute-agents/fleet/cpu-run -c 4 --mem 16G -t 60 bash cluster/qwen_jax_env.sh
# The venv lands at $QWEN_JAX_DIR/param_decomp/.venv; call its python directly (`uv run` would re-sync without CUDA).
set -euo pipefail
export QWEN_JAX_DIR=${QWEN_JAX_DIR:-$HOME/commute-data/param-decomp-177d0cce8}
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=$HOME/commute-data/uv-cache-qwen-jax   # the worker cannot see the dev node's /ephemeral
export UV_LINK_MODE=copy

if [[ ! -d $QWEN_JAX_DIR ]]; then
    git clone https://github.com/goodfire-ai/param-decomp.git "$QWEN_JAX_DIR"
fi
git -C "$QWEN_JAX_DIR" checkout 177d0cce8
cd "$QWEN_JAX_DIR"
uv sync --project param_decomp --frozen --no-dev --extra cuda   # CUDA 12.8 wheels (pre-r580 driver arm)
param_decomp/.venv/bin/python -c "import jax, orbax.checkpoint, tensorstore, safetensors; print('jax', jax.__version__)"
rm -rf "$UV_CACHE_DIR"
