# Source this on the MATS cluster (controller or inside a Slurm job): `source cluster/env.sh`.
# Code, venv and results live on NFS; caches and temp files on the node's disposable /ephemeral.
# Never point anything at /tmp (MATS cluster handbook).

export COMMUTE_REPO=/mnt/nw/home/e.agnew/commute
_eph=/ephemeral/$USER

export TMPDIR=$_eph/tmp
export HF_HOME=$_eph/hf                    # tokenizer + datasets cache (re-downloadable); overrides env.py's artifacts/hf default
export PIP_CACHE_DIR=$_eph/pip
export XDG_CACHE_HOME=$_eph/cache
export UV_CACHE_DIR=$_eph/uv
export UV_LINK_MODE=copy                   # cache (/ephemeral) and venv (NFS) are different filesystems
export TRITON_CACHE_DIR=$_eph/triton
export TORCHINDUCTOR_CACHE_DIR=$_eph/torchinductor
export MPLCONFIGDIR=$_eph/mpl
export JUPYTER_RUNTIME_DIR=$_eph/jupyter-runtime
mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$JUPYTER_RUNTIME_DIR"
unset _eph

export PATH=$HOME/.local/bin:$PATH        # uv lives in ~/.local/bin (NFS home)
export UV_NO_SYNC=1                       # a bare `uv run` would swap in the default (CPU) torch; rebuild via setup.sbatch
