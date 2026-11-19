"""Where the released VPD decomposition lives on W&B, and its download. Imports no torch, so a fresh pod can fetch
while `uv sync` builds the venv: `PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.vpd_files`.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from commute import env

DECOMP_RUN = "s-55ea3f9b"
TARGET_RUN = "t-9d2b8f02"
WANDB_PROJECT = "goodfire/spd"
DECOMP_FILES = ("model_400000.pth", "final_config.yaml")
TARGET_FILES = ("final_config.yaml", "model_config.yaml")
TARGET_CKPT = "model_step_99999.pt"

# Slow-host guard for `python -m commute.vpd_files` (cluster/pod_setup.sh). Some fresh pods fetch from W&B at
# 0.3-0.4 MB/s instead of 30-45 MB/s; the 2.9 GB decomposition then takes ~2 h while the GPU idles (friction 853, 862,
# 863, 871: ~$30 and hours on E203 and E208). After CHECK_S seconds of one file's download, a mean rate under
# MIN_MB_S exits the process with SLOW_EXIT, and pod_setup.sh tells the runner to send the files with sync() instead.
SLOW_EXIT = 75
MIN_MB_S = float(os.environ.get("VPD_FETCH_MIN_MB_S", "5"))
CHECK_S = float(os.environ.get("VPD_FETCH_CHECK_S", "60"))


def _rate_guard(scratch: Path, size: int, label: str, min_mb_s: float, check_s: float, log: Any,
                done: threading.Event) -> None:
    """Watch the bytes written under `scratch`; once `check_s` seconds have passed, exit the whole process with
    SLOW_EXIT if the mean rate is under `min_mb_s` (os._exit: the W&B download thread cannot be cancelled)."""
    t0 = time.time()
    if done.wait(check_s):
        return
    got = sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())
    rate = got / 1e6 / (time.time() - t0)
    if rate < min_mb_s:
        log(f"[fetch] {label}: {got / 1e6:.1f} of {size / 1e6:.1f} MB in {time.time() - t0:.0f} s = {rate:.2f} MB/s,"
            f" under {min_mb_s:g} MB/s: giving up on W&B (exit {SLOW_EXIT}); send the files with sync() instead")
        sys.stdout.flush()
        os._exit(SLOW_EXIT)
    log(f"[fetch] {label}: {rate:.1f} MB/s after {check_s:.0f} s, continuing")


def fetch(log: Any = print, min_mb_s: float | None = None, check_s: float = CHECK_S) -> None:
    """Download the two runs once. A file already in place is kept without asking W&B (downloads land whole: each is
    moved into place only once complete), so a pod sent the files needs no W&B access. Downloading needs
    WANDB_API_KEY in the environment; never prints it. With `min_mb_s`, a file still downloading after `check_s`
    seconds at a mean rate under `min_mb_s` MB/s ends the process with exit code SLOW_EXIT (only the command-line
    entry point sets it, so a library caller is never killed)."""
    import wandb

    wanted: list[tuple[str, str, str]] = [(DECOMP_RUN, f, f) for f in DECOMP_FILES]
    wanted += [(TARGET_RUN, f, f) for f in TARGET_FILES] + [(TARGET_RUN, TARGET_CKPT, f"ckpt/{TARGET_CKPT}")]
    api = None
    for run_id, name, rel in wanted:
        dest = env.VPD_DIR / run_id / rel
        if dest.is_file():
            log(f"[fetch] {run_id}/{rel}: present ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        if api is None:
            if not os.environ.get("WANDB_API_KEY"):
                raise RuntimeError("WANDB_API_KEY is not set in the environment")
            api = wandb.Api()
        f = api.run(f"{WANDB_PROJECT}/{run_id}").file(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        scratch = dest.parent / f"_dl_{dest.name}"
        scratch.mkdir(exist_ok=True)
        done = threading.Event()
        if min_mb_s is not None:
            args = (scratch, int(f.size), f"{run_id}/{rel}", min_mb_s, check_s, log, done)
            threading.Thread(target=_rate_guard, args=args, daemon=True).start()
        try:
            f.download(root=str(scratch), replace=True)
        finally:
            done.set()
        os.replace(scratch / name, dest)
        scratch.rmdir()
        log(f"[fetch] {run_id}/{rel}: {int(f.size) / 1e6:.1f} MB in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    fetch(log=lambda m: print(m, flush=True), min_mb_s=MIN_MB_S)
