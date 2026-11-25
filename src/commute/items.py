"""Saved updates by one naming scheme, so a reading loads a state without another experiment's run.py (the reading
refactor, P2.79 move 1; plan reports/plans/2026-11-25_plan_reading-refactor.md, appendix A3).

An Item is `<family>/<task>_s<seed>[_scaled[_<pair>]]`, the file of its delta-W over the 24 decomposed matrices, the
file's sha256 where a record holds one. `load_dw` gives fp32 CPU deltas
keyed by module (`h.0.mlp.down_proj`), as `steering.Net.at` and every `realistic_reads` reading take them.

Families (`inventory`):
  e136   E136's inventory states (experiments/share_untrained/results/inventory.json): e46/*, e54/*, c3/*, e62/*,
         case_label/*, fresh/*, e53/*, class/* (no sha256 was recorded for them)
  p256   compare-lead's 45 P2.56 states (experiments/rb_states/results/manifest_p256.json, sha256 recorded)
Old names still resolve (`resolve`): E98's bare `<task>_s<seed>` where unique in E136's inventory, and the file-name
form `<family>-<task>_s<seed>` that kstar_needed, claim2_needed and rb_states write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache

import torch
from torch import Tensor

from commute.jobs import EXPERIMENTS
from commute.windows import REPO, file_sha256

DW = dict[str, Tensor]
N_MATRICES = 24


@dataclass(frozen=True)
class Item:
    name: str  # <family>/<task>_s<seed>[_scaled[_<pair>]]
    path: str  # repo path of the delta-W file
    sha256: str | None  # of the file, where a record holds one


def load_dw(item: Item) -> DW:
    """The item's delta-W as fp32 CPU tensors keyed by module, sha256 checked where recorded."""
    p = REPO / item.path
    if item.sha256 is not None and (got := file_sha256(p)) != item.sha256:
        raise SystemExit(f"STOP: {item.path}: sha256 {got}, the record's {item.sha256}")
    d = torch.load(p, weights_only=True, map_location="cpu")
    assert len(d) == N_MATRICES and all(k.startswith("h.") and not k.endswith(".weight") for k in d), f"{item.path}: not 24 module-keyed matrices"
    return {k: v.float() for k, v in d.items()}


@cache
def inventory(family: str) -> dict[str, Item]:
    if family == "e136":
        states = json.loads((EXPERIMENTS / "share_untrained" / "results" / "inventory.json").read_text())["states"]
        return {n: Item(n, s["file"], None) for n, s in states.items()}
    if family == "p256":
        man = json.loads((EXPERIMENTS / "rb_states" / "results" / "manifest_p256.json").read_text())
        return {e["item"]: Item(e["item"], e["dw"], e["sha256"]) for e in man}
    raise KeyError(f"no item family {family!r} (e136, p256)")


FAMILIES = ("e136", "p256")


@cache
def _aliases() -> dict[str, str]:
    """Old name -> item name: bare E98 names unique in E136's inventory, and every item's file-name form."""
    every = [n for f in FAMILIES for n in inventory(f)]
    bare: dict[str, list[str]] = {}
    for n in inventory("e136"):
        bare.setdefault(n.split("/", 1)[1], []).append(n)
    return {b: ns[0] for b, ns in bare.items() if len(ns) == 1} | {n.replace("/", "-"): n for n in every}


def resolve(name: str) -> Item:
    """An item by its name or an old name."""
    for f in FAMILIES:
        if name in inventory(f):
            return inventory(f)[name]
    full = _aliases().get(name)
    if full is None:
        raise KeyError(f"no item {name!r}")
    return resolve(full)

