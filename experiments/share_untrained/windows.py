"""E136's read windows, one loader per format of `results/inventory.json`. Each is the path its own experiment's read took: a kv task's held-out
windows (`steering.load_task`, the table as target), E62's read windows with each map's target as a last column (`copying.with_target`, data.LAST, ci_len the
window's length), the label pairs (E133's `prior_vs_share.run.label_windows`), and E53's scored Pile windows (E60's `e53_tasks`, the target each window's next token)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import copying as CP
from commute.data import LAST, NEXT
from commute.jobs import load_experiment

REPO = Path(__file__).resolve().parents[2]
_e53: dict[str, Any] = {}


@dataclass(frozen=True)
class Read:
    name: str
    toks: Tensor  # (N, T), any target column last
    half: int  # the answer position
    token: int | Tensor  # a (V,) table, data.LAST or data.NEXT
    batch: int  # the batch of the pair's own route read
    ci_len: int | None


def _e53_tasks() -> tuple[dict[str, Any], int]:
    """E60's loader, once: E53's scored windows per task, built from the cached Pile val rows."""
    if not _e53:
        TT = load_experiment("regating_truth_table", "run")
        pa = TT.performed_anchor()
        _e53["tasks"], _e53["half"] = TT.e53_tasks(pa, pa.Run.__new__(pa.Run)), pa.CFG.half  # `scored` reads no attribute of Run
    return _e53["tasks"], _e53["half"]


def load_read(name: str, spec: dict[str, Any]) -> Read:
    """One task's read, on CPU."""
    fmt, batch = spec["format"], spec["batch"]
    if fmt == "kv":
        w = torch.load(REPO / spec["path"], weights_only=False)
        return Read(name, w["heldout"].tokens, w["heldout"].half, w["table"], batch, None)
    if fmt == "windows_pt":
        w = torch.load(REPO / spec["path"], weights_only=False)
        read = w["read"]
        return Read(name, CP.with_target(read, w["target"][spec["arm"]]["read"]).tokens, read.half, LAST, batch, read.tokens.shape[1])
    if fmt == "label_task":
        pvs = load_experiment("prior_vs_share", "run")
        return Read(name, pvs.label_windows(spec), pvs.HALF, LAST, batch, pvs.LEN)
    assert fmt == "e53", fmt
    tasks, half = _e53_tasks()
    return Read(name, tasks[spec["task"]].held, half, NEXT, batch, None)
