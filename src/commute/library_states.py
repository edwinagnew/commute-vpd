"""The Qwen3-1.7B pair library's saved states (cell P3.5): which W&B artifact holds an arm's save, and its fetch.

Each state is δW = W_t - W0, fp32 safetensors, 196 tensors keyed by site (experiments/SAVED.md). The library's per-arm
JSON (experiments/qwen_pair_library/results/<run>/<arm>_s<seed>.json) records every save: its labels, step, size,
sha256 and artifact. Run 2 (E164) is `library`, artifacts `p35-<arm>-s<seed>-<labels>`; run 3 (E165) is
`library_run3`, artifacts `p35r3-...`. A save is found by its label in `.saves[].labels`, since one state can carry two
labels ("5x_15x"); the file is `<joined labels>.safetensors`. No torch here: a fresh pod can fetch while its venv builds.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from commute import qwen_files
from commute.jobs import EXPERIMENTS

LIBRARY = EXPERIMENTS / "qwen_pair_library" / "results"
PREFIX = {"library": "p35", "library_run3": "p35r3"}  # the library's results folder -> its W&B artifact type


@dataclass(frozen=True)
class State:
    run: str  # "library" (run 2) or "library_run3" (run 3)
    arm: str
    seed: int
    save: str  # the label asked for: "5x", "15x" or "last"
    labels: str  # every label of the state that holds it, joined with "_"
    step: int
    bytes: int
    sha256: str
    artifact: str
    dw_norm: float  # |δW| as the library read it from the file

    @property
    def key(self) -> str:
        return f"{self.arm}_s{self.seed}_{self.save}"

    @property
    def file(self) -> str:
        return f"{self.labels}.safetensors"


def state(run: str, arm: str, seed: int, save: str) -> State:
    """The library's record of the save labelled `save` of one arm and seed (exactly one state carries each label)."""
    arm_json: dict[str, Any] = json.loads((LIBRARY / run / f"{arm}_s{seed}.json").read_text())
    hits = [s for s in arm_json["saves"] if save in s["labels"]]
    assert len(hits) == 1, f"{run}/{arm}_s{seed}: {len(hits)} saves labelled {save}"
    s, labels = hits[0], "_".join(hits[0]["labels"])
    assert s["artifact"] == f"{PREFIX[run]}-{arm}-s{seed}-{labels}", s["artifact"]
    return State(run, arm, seed, save, labels, s["step"], s["bytes"], s["sha256"], s["artifact"],
                 arm_json["states"][labels]["dw_norm"])


def fetch(st: State, root: Path, log: Any = print) -> Path:
    """The state's file into root/<key>/ from W&B (kept if already whole), checked against the library's sha256; a
    mismatch raises. Needs WANDB_API_KEY unless the file is there."""
    dest = root / st.key
    t0 = time.time()
    qwen_files.fetch(dest, log, artifact=st.artifact, kind=PREFIX[st.run],
                     ok=lambda d: (d / st.file).is_file() and (d / st.file).stat().st_size == st.bytes)
    got = sha256(dest / st.file)
    log(f"[fetch {st.key}] {st.artifact}: {time.time() - t0:.0f} s, sha256 {'matches' if got == st.sha256 else 'DIFFERS'}")
    if got != st.sha256:
        raise ValueError(f"{st.artifact}: sha256 {got}, the library recorded {st.sha256}")
    return dest / st.file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()
