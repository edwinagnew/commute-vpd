# Source this on the MATS cluster (controller or inside a Slurm job): `source cluster/env.sh`.
# Code, venv and results live on NFS; caches and temp files on the node's disposable /ephemeral.
# Never point anything at /tmp (MATS cluster handbook).

export COMMUTE_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # the clone this file is in
_eph=/ephemeral/${USER:-$(id -un)}   # a restarted lead's shell can lack USER (job 39125 died on it, 2026-10-27)

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
export PYTHONPATH=$COMMUTE_REPO/src${PYTHONPATH:+:$PYTHONPATH}   # this clone's code: the shared venv's commute.pth names whichever clone synced last
export UV_NO_SYNC=1                       # a bare `uv run` would swap in the default (CPU) torch; rebuild via setup.sbatch
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # E45 P OOMed on an L40 with 13 GB reserved but unallocated (friction 108)
export PYTHONUNBUFFERED=1                 # a job's .out shows progress as it runs, not only when the job ends (friction 8)
export COMMUTE_QWEN_DIR=/mnt/nw/home/e.agnew/commute-data/artifacts/qwen3-1.7b-p-e9fcffd5-torch   # Qwen3-1.7B decomposition as fp32 safetensors (experiments/qwen_port/convert.py; index.json)
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  # In the job's .out before any Python: silence after this line is a hang in Python's start-up (friction 132). In jobs.log
  # because a job's node and GPU cannot be asked for after it ends: no sacct, and scontrol forgets it (friction 200).
  # To standard error, since a caller may parse a job's output; never fatal, since jobs run under set -e.
  _started="$(date -Is) $SLURM_JOB_ID ${SLURM_JOB_NAME:-} started node=${SLURMD_NODENAME:-?} gpus=${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-none}}"   # a direct srun sets only the step's
  echo "$_started" >&2 || true
  { echo "$_started" >> "$COMMUTE_REPO/cluster/logs/jobs.log"; } 2>/dev/null || true
  unset _started
  # Slurm hands out a card held by processes it does not count: jobs then died at load with a bare CUDA out of memory, and
  # an afterok chain burnt through on the same card in 90 s (friction 445, 446, 448, 470, 474). Say so and stop first. Once
  # per job (the marker), since a later source in the same job sees the job's own memory. Exit 75: nothing ran.
  if [[ -n ${CUDA_VISIBLE_DEVICES:-} && ! -e $TMPDIR/gpu-checked-$SLURM_JOB_ID ]] && command -v nvidia-smi > /dev/null; then
    : > "$TMPDIR/gpu-checked-$SLURM_JOB_ID" 2> /dev/null || true
    _used=$(nvidia-smi --id="${SLURM_JOB_GPUS:-$CUDA_VISIBLE_DEVICES}" --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | sort -n | tail -1) || _used=
    echo "$(date -Is) $SLURM_JOB_ID allocated GPU memory used: ${_used:-unknown} MiB" >&2 || true
    if [[ $_used =~ ^[0-9]+$ ]] && (( _used > ${COMMUTE_GPU_MAX_USED_MB:-5000} )); then
      echo "cluster/env.sh: GPU ${SLURM_JOB_GPUS:-$CUDA_VISIBLE_DEVICES} already holds $_used MiB outside this job; stopping before loading (exit 75). COMMUTE_GPU_MAX_USED_MB overrides." >&2
      { echo "$(date -Is) $SLURM_JOB_ID stopped: GPU already held ($_used MiB)" >> "$COMMUTE_REPO/cluster/logs/jobs.log"; } 2> /dev/null || true
      exit 75
    fi
    unset _used
  fi
fi
