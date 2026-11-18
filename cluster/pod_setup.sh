#!/usr/bin/env bash
# Run on a fresh pod (RunPod or Vast), from the synced tree (the kernel's cwd), once after the first sync():
#
#   bash cluster/pod_setup.sh
#
# Pinned uv, Python 3.13, the CUDA venv (.venv) and the VPD files, with the VPD fetch running alongside `uv sync`.
# Each step prints "[setup HH:MM:SS +Ns] ..." to /root/setup.log and the RunPod console, so the set-up time is
# measured on every pod. Everything here lands on the container disk (remote-kernels.toml: volume-gb = 0, workdir
# /root/work), never on RunPod's network storage.
#
# Filed predictions on a pod: sync() sends no .git, so `commute.jobs.require_filed` cannot ask git whether
# predictions.json is committed unchanged. It compares the file's sha256 with PRED_SHA instead (E120_PRED_SHA, the old
# name, is still read), and refuses (exit 1) when that is unset or different: all six E142 pods exited 1 on first
# launch for want of it (friction 599, 680). When PRED_SHA is set it is checked even if a .git exists (friction 756,
# 767: a clone step left a .git and git refused). The brief computes it on the dev node from the committed file and
# the run line on the pod carries it:
#   git show HEAD:experiments/<name>/results/predictions.json | sha256sum     # on the dev node
#   PRED_SHA=<that sha256> .venv/bin/python experiments/<name>/run.py ...      # on the pod
# A .git with no commit (a clone step whose fetch failed, e.g. GIT_PAT unset) is removed below.
#
# Settings, as environment variables of this script:
#   POD_IDLE_MIN  idle-GPU minutes before the pod terminates itself (default 30; cluster/pod_idle_watchdog.sh). An arm
#                 that must stay up while a sibling arm runs sets it from the plan's wall time up front, e.g.
#                 `POD_IDLE_MIN=600 bash cluster/pod_setup.sh` (friction 763); re-arming by hand later is the fallback.
# Defaults this script sets for every later `.venv/bin/python` on the pod (a .pth file in the pod's own venv):
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unless already set (friction 776: E178 CUDA OOM from fragmentation).
# Lint: pyright's nodeenv cannot install node on pods (it parses a "0-alpha" version), so the pod venv gets the
# nodejs-wheel-binaries package, which pyright uses instead of nodeenv (friction 788, 796).
set -Eeuo pipefail  # -E: the ERR trap also fires in the fetch subshell
UV_VERSION=0.12.9  # the laptop's; 0.5.11 does not know --no-default-groups
exec > >(tee -a /root/setup.log /proc/1/fd/1) 2>&1
t0=$SECONDS
step() { echo "[setup $(date -u +%T) +$((SECONDS - t0))s] $*"; }
trap 'step "FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR

source cluster/pod_env.sh  # Vast: the kernel that runs this script lacks the machine's variables
have() { [[ -n "${!1:-}" ]] && echo set || echo MISSING; }  # presence only, never the value
step "start in $PWD"
if [[ -d .git ]] && ! git rev-parse -q --verify HEAD > /dev/null 2>&1; then  # friction 756, 767
  rm -rf .git; step "removed a .git with no commit (clone fetch failed or GIT_PAT $(have GIT_PAT))"
fi
# First, so it covers set-up too: the pod terminates itself after POD_IDLE_MIN idle GPU minutes (default 30).
export POD_IDLE_MIN=${POD_IDLE_MIN:-30}
setsid nohup bash cluster/pod_idle_watchdog.sh > /dev/null 2>&1 < /dev/null &
if [[ -n "${CONTAINER_ID:-}" ]]; then step "idle watchdog started (POD_IDLE_MIN=$POD_IDLE_MIN); vast CONTAINER_ID set, CONTAINER_API_KEY $(have CONTAINER_API_KEY)"
else step "idle watchdog started (POD_IDLE_MIN=$POD_IDLE_MIN); runpodctl $(command -v runpodctl > /dev/null && echo present || echo MISSING)," \
     "RUNPOD_POD_ID $(have RUNPOD_POD_ID), RUNPOD_API_KEY $(have RUNPOD_API_KEY)"; fi
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
site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')" > "$site/commute_pod_env.pth"
step "PYTORCH_CUDA_ALLOC_CONF default expandable_segments:True (via $site/commute_pod_env.pth)"
if uv pip install -q --python .venv/bin/python nodejs-wheel-binaries==22.20.0; then step "node for pyright (nodejs-wheel-binaries)"
else step "WARNING: nodejs-wheel-binaries did not install, so make lint's pyright step will fail here; ruff still runs"; fi

.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
[[ -e .git ]] || step "no .git: a run's require_filed needs PRED_SHA=<sha256 of the committed predictions.json> (see the top of this file)"
step "done"
