#!/usr/bin/env bash
# Run on a fresh RunPod POD, from the synced tree (the kernel's cwd), once after the first sync():
#
#   bash cluster/pod_setup.sh
#
# Pinned uv, Python 3.13, the CUDA venv (.venv) and the VPD files, with the VPD fetch running alongside `uv sync`.
# Each step prints "[setup HH:MM:SS +Ns] ..." to /root/setup.log and the RunPod console, so the set-up time is
# measured on every pod. Everything here lands on the container disk (remote-kernels.toml: volume-gb = 0, workdir
# /root/work), never on RunPod's network storage.
set -Eeuo pipefail  # -E: the ERR trap also fires in the fetch subshell
UV_VERSION=0.12.9  # the laptop's; 0.5.11 does not know --no-default-groups
exec > >(tee -a /root/setup.log /proc/1/fd/1) 2>&1
t0=$SECONDS
step() { echo "[setup $(date -u +%T) +$((SECONDS - t0))s] $*"; }
trap 'step "FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR

step "start in $PWD"
# First, so it covers set-up too: the pod terminates itself after POD_IDLE_MIN idle GPU minutes (default 30).
setsid nohup bash cluster/pod_idle_watchdog.sh > /dev/null 2>&1 < /dev/null &
have() { [[ -n "${!1:-}" ]] && echo set || echo MISSING; }  # presence only, never the value
step "idle watchdog started; runpodctl $(command -v runpodctl > /dev/null && echo present || echo MISSING)," \
     "RUNPOD_POD_ID $(have RUNPOD_POD_ID), RUNPOD_API_KEY $(have RUNPOD_API_KEY)"
[[ "$(uv --version 2>/dev/null)" == "uv $UV_VERSION"* ]] || pip install -q --root-user-action=ignore "uv==$UV_VERSION"
step "$(uv --version)"
uv python install -q 3.13
step "python 3.13"

(PYTHONPATH=src uv run -q --no-project --python 3.13 --with wandb==0.23.1 python -m commute.vpd_files
 step "VPD files") &
fetch=$!
uv sync -q --locked --no-default-groups --group dev --group gpu
step "uv sync"
wait "$fetch"

.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
step "done"
