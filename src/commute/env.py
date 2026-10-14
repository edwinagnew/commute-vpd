"""Cache root and the environment variables that must be set before `param_decomp`, `datasets`
or `huggingface_hub` are imported (param_decomp mkdirs its output dir at import time).

`COMMUTE_CACHE_DIR` is the root: `<repo>/artifacts` locally (gitignored). Beneath it `vpd/` holds
the two W&B runs, `hf/` the HuggingFace cache, `out/` param_decomp's output dir.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CACHE_DIR: Path = Path(os.environ.get("COMMUTE_CACHE_DIR", REPO_ROOT / "artifacts")).resolve()
VPD_DIR: Path = CACHE_DIR / "vpd"
HF_DIR: Path = CACHE_DIR / "hf"
OUT_DIR: Path = CACHE_DIR / "out"

os.environ.setdefault("PARAM_DECOMP_OUT_DIR", str(OUT_DIR))
os.environ.setdefault("HF_HOME", str(HF_DIR))
for _d in (VPD_DIR, HF_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
