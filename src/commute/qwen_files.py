"""Where the converted Qwen3-1.7B decomposition lives on W&B, and its download (uploaded by
`experiments/qwen_port/upload.py`). Imports no torch, so a fresh pod can fetch while `uv sync` builds the venv:
  PYTHONPATH=src uv run --no-project --with wandb==0.23.1 python -m commute.qwen_files /workspace/qwen
then point `COMMUTE_QWEN_DIR` at that directory (`commute.qwen_vpd` reads it).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ENTITY = "edwinagnew-mats"
PROJECT = "commute"
ARTIFACT = "qwen3-1.7b-p-e9fcffd5-torch"
ARTIFACT_TYPE = "decomposition"


def complete(dest: Path) -> bool:
    """True when `index.json`, `launch_config.yaml` and every shard index.json lists are at `dest`, each shard with
    the manifest's size. Files land whole (moved into place once downloaded), so a right-sized file is a full one."""
    index = dest / "index.json"
    if not (index.is_file() and (dest / "launch_config.yaml").is_file()):
        return False
    files = json.loads(index.read_text())["files"]
    return all((dest / n).is_file() and (dest / n).stat().st_size == f["bytes"] for n, f in files.items())


def fetch(dest: Path, log: Any = print, artifact: str = ARTIFACT, kind: str = ARTIFACT_TYPE,
          ok: Callable[[Path], bool] = complete, version: str = "latest") -> None:
    """Fill `dest` with `artifact` at `version` (default the decomposition, latest; `ok` says when `dest` is complete). A complete `dest`
    is kept without asking W&B, so a pod on the network volume needs no W&B access. Otherwise only the files missing
    or of the wrong size are downloaded, each into a scratch directory beside `dest` and moved into place once whole.
    Downloading needs WANDB_API_KEY; never prints it."""
    if ok(dest):
        log(f"[fetch] {dest}: complete")
        return
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set in the environment")
    import wandb

    art = wandb.Api().artifact(f"{ENTITY}/{PROJECT}/{artifact}:{version}", type=kind)
    scratch = dest.parent / f"_dl_{dest.name}"
    dest.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(exist_ok=True)
    pool = ThreadPoolExecutor(16)  # parts of one file in parallel, as wandb's own Artifact.download does
    for name, entry in sorted(art.manifest.entries.items()):
        if (dest / name).is_file() and (dest / name).stat().st_size == entry.size:
            continue
        t0 = time.time()
        entry.download(root=str(scratch), skip_cache=True, executor=pool)
        (dest / name).parent.mkdir(parents=True, exist_ok=True)  # a name may hold a directory (dW/step_1...)
        os.replace(scratch / name, dest / name)
        log(f"[fetch] {name}: {(entry.size or 0) / 1e6:.0f} MB in {time.time() - t0:.0f} s")
    pool.shutdown()
    shutil.rmtree(scratch)
    assert ok(dest), f"{dest} incomplete after download of {art.qualified_name}"


if __name__ == "__main__":
    fetch(Path(sys.argv[1]))
