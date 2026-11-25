"""Saved updates by one naming scheme, so a reading loads a state without another experiment's run.py (the reading
refactor, P2.79 move 1; plan reports/plans/2026-11-25_plan_reading-refactor.md, appendix A3).

An Item is `<family>/<task>_s<seed>[_scaled[_<pair>]]`, the file of its delta-W over the 24 decomposed matrices, the
file's sha256 where a record holds one, and the factor it is read at (None: as saved). `load_dw` gives fp32 CPU deltas
keyed by module (`h.0.mlp.down_proj`), as `steering.Net.at` and every `realistic_reads` reading take them.

Families (`inventory`):
  e136   E136's inventory states (experiments/share_untrained/results/inventory.json): e46/*, e54/*, c3/*, e62/*,
         case_label/*, fresh/*, e53/*, class/* (no sha256 was recorded for them)
  p256   compare-lead's 45 P2.56 states (experiments/rb_states/results/manifest_p256.json, sha256 recorded)
Old names still resolve (`resolve`): E98's bare `<task>_s<seed>` where unique in E136's inventory, and the file-name
form `<family>-<task>_s<seed>` that kstar_needed, claim2_needed and rb_states write.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch import Tensor

from commute import old_tasks as OT
from commute import rule_retention as R
from commute import variable_read as VR
from commute import vpd
from commute.jobs import EXPERIMENTS
from commute.windows import REPO, file_sha256

DW = dict[str, Tensor]
N_MATRICES = 24


@dataclass(frozen=True)
class Item:
    name: str  # <family>/<task>_s<seed>[_scaled[_<pair>]]
    path: str  # repo path of the delta-W file
    sha256: str | None  # of the file, where a record holds one
    scale: float | None  # the factor the update is read at; None reads it as saved
    pair: str | None  # the pair whose size match set `scale` (E232's form), None otherwise


def load_dw(item: Item) -> DW:
    """The item's delta-W as fp32 CPU tensors keyed by module, sha256 checked where recorded, times `scale` if set."""
    p = REPO / item.path
    if item.sha256 is not None and (got := file_sha256(p)) != item.sha256:
        raise SystemExit(f"STOP: {item.path}: sha256 {got}, the record's {item.sha256}")
    d = torch.load(p, weights_only=True, map_location="cpu")
    assert len(d) == N_MATRICES and all(k.startswith("h.") and not k.endswith(".weight") for k in d), f"{item.path}: not 24 module-keyed matrices"
    return {k: v.float() * item.scale if item.scale is not None else v.float() for k, v in d.items()}


@cache
def inventory(family: str) -> dict[str, Item]:
    if family == "e136":
        states = json.loads((EXPERIMENTS / "share_untrained" / "results" / "inventory.json").read_text())["states"]
        return {n: Item(n, s["file"], None, None, None) for n, s in states.items()}
    if family == "p256":
        man = json.loads((EXPERIMENTS / "rb_states" / "results" / "manifest_p256.json").read_text())
        return {e["item"]: Item(e["item"], e["dw"], e["sha256"], None, None) for e in man}
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


# ----------------------------------------------------------------------------- items built with their F, M and D
# The fresh write passes and the neuron K scan (P2.79 2c) need F itself, not a saved delta-W: E214's built x<f> (a
# rescale of M) and E229's trained items (M plus an old task's delta-W), with their D and E217's neuron bounds. Copied
# from the finished run.py files that built them (kstar_levels `World.make`, rule_retention `rescaled` / COPY24,
# topk_neurons `setup` / `windows` / `pair`, rule_retention_pairs `Data.read_half` / `subset` / `fine_tune`, topk_upper
# `load_pct` / `feat_of`), which stay as they ran. Every setting is an argument; callers take them from those files'
# predictions.json "settings".

COPY24 = EXPERIMENTS / "open_pair" / "results" / "large" / "amend1" / "windows_copy_trained.pt"  # E195's copy24 windows
E217_PCT = EXPERIMENTS / "topk_neurons" / "results" / "large" / "percentiles.pt"


def rescaled(M: R.Model, layer: int, C: Sequence[int], factor: float) -> R.Model:
    """E214's x<factor>: a copy of M with block `layer`'s down_proj columns C times `factor`."""
    F = R.Model(copy.deepcopy(M.net).requires_grad_(False))
    with torch.no_grad():
        F.net.h[layer].mlp.down_proj.weight[:, list(C)] *= factor
    return F


def copy24_held() -> Tensor:
    """E214's D of x<f>: E195's copy24 held-out windows (CPU), read at every position."""
    return torch.load(COPY24, weights_only=False)["heldout"].tokens


def e229_read(task: str, read_seed: int, n_read: int) -> Tensor:
    """E229's D of an old task's items: n_read windows of the task's read half (`old_tasks.task_windows`), drawn by
    read_seed (CPU), read at the last position."""
    sizes = OT.DSizes()
    read = OT.task_windows(OT.TASKS[task], VR.make_sets(sizes, vpd.tokenizer(), print), sizes)[1]
    return read[torch.randperm(read.shape[0], generator=torch.Generator().manual_seed(read_seed))[:n_read]]


def e229_f(net: Any, item: str) -> R.Model:
    """E229's F of `<task>_s<seed>`: M (net, a `steering.Net`) plus the old task's saved delta-W (`old_tasks.old_f`)."""
    task, seed = item.rsplit("_s", 1)
    return R.Model(OT.old_f(net, OT.TASKS[task].states[int(seed)]).net)


def neuron_bounds(q: float, qs: Sequence[float], sha256: str) -> Tensor:
    """E217's per-neuron q-quantile of M's MLP activations over Pile rows, (layers, n) CPU: E229's upper bounds (an
    activation above it is a feature's value). The saved file is checked against `sha256` and its quantiles against qs."""
    if (got := file_sha256(E217_PCT)) != sha256:
        raise SystemExit(f"STOP: {E217_PCT}: sha256 {got}, the filed {sha256}")
    z = torch.load(E217_PCT, weights_only=True)
    assert torch.equal(z["q"], torch.tensor(list(qs))), "the saved quantiles are not the filed ones"
    return z["values"][list(qs).index(q)]
