"""Where the released VPD decomposition lives on W&B, and its download. Imports no torch, so a fresh pod can fetch
while `uv sync` builds the venv: `PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.vpd_files`.
"""

from __future__ import annotations

import os
import time
from typing import Any

from commute import env

DECOMP_RUN = "s-55ea3f9b"
TARGET_RUN = "t-9d2b8f02"
WANDB_PROJECT = "goodfire/spd"
DECOMP_FILES = ("model_400000.pth", "final_config.yaml")
TARGET_FILES = ("final_config.yaml", "model_config.yaml")
TARGET_CKPT = "model_step_99999.pt"


def fetch(log: Any = print) -> None:
    """Download the two runs once. Needs WANDB_API_KEY in the environment; never prints it."""
    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set in the environment")
    api = wandb.Api()
    wanted: list[tuple[str, str, str]] = [(DECOMP_RUN, f, f) for f in DECOMP_FILES]
    wanted += [(TARGET_RUN, f, f) for f in TARGET_FILES] + [(TARGET_RUN, TARGET_CKPT, f"ckpt/{TARGET_CKPT}")]
    for run_id, name, rel in wanted:
        dest = env.VPD_DIR / run_id / rel
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        f = run.file(name)
        if dest.is_file() and dest.stat().st_size == int(f.size):
            log(f"[fetch] {run_id}/{rel}: present ({int(f.size) / 1e6:.1f} MB)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        scratch = dest.parent / f"_dl_{dest.name}"
        scratch.mkdir(exist_ok=True)
        f.download(root=str(scratch), replace=True)
        os.replace(scratch / name, dest)
        scratch.rmdir()
        log(f"[fetch] {run_id}/{rel}: {int(f.size) / 1e6:.1f} MB in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    fetch()
