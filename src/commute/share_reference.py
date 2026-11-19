"""Shared steps of the re-gating share's reference on the frozen-neuron decomposition p-015e0710 (cells P3.10 and its
repair): P3.9's records (the fine-tunes' sizes, the row sets' floors, the random changes' norms), the built change's
neuron orders and its weight change, and one random draw's reads at the three sizes. First written in
experiments/qwen_share_reference/run.py (E181), which stays as it ran; the size search, the reads and the placement
rule are in `commute.share_reading`.

The built change here takes a continuous size x: the gate_proj rows of the first floor(x) neurons of an order are
zeroed and the next neuron's row is multiplied by 1 - frac(x). E181's whole count k is x = k. Each neuron is one
subcomponent of p-015e0710, so the change is a mask on each subcomponent it touches: 0 on the first floor(x), 1 -
frac(x) on the next.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import share_reading as SR
from commute.jobs import EXPERIMENTS
from commute.qwen_vpd import N_LAYERS, site_name

FROZEN = "p-015e0710"
P39 = EXPERIMENTS / "qwen_library_share" / "results" / FROZEN  # P3.9: run1 (random norms), run2 (fine-tunes), floors
ARMS_OF = {"copy": ("copy",), "cipher": ("cipher",), "known": ("known",), "invented": ("invented",),
           "gold": ("gold", "gold_shuf")}  # the library read both gold arms on gold's rows
SEEDS = (0, 1, 2)  # P3.9's fine-tune seeds
SAVES = ("5x", "last")
CHOICES = ("top", "random", "specific")
LEVELS = ("low", "mid", "high")


def stored(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


# ----------------------------------------------------------------------------- P3.9's files: sizes, floors, norms


def finetunes(rowset: str) -> list[dict[str, Any]]:
    """P3.9's read fine-tunes on this row set (run2: arms x seeds 0-2 x saves 5x and last)."""
    recs = [json.loads((P39 / "run2" / f"{a}_s{s}_{v}.json").read_text())
            for a in ARMS_OF[rowset] for s in SEEDS for v in SAVES]
    assert all(r["state"]["arm"] in ARMS_OF[rowset] for r in recs)
    return recs


def sizes(rowset: str) -> dict[str, float]:
    """The three target K00s: the row set's fine-tunes' lowest, their geometric middle and their highest."""
    k = [r["means"]["K00"] for r in finetunes(rowset)]
    return {"low": min(k), "mid": math.sqrt(min(k) * max(k)), "high": max(k)}


def floor(rowset: str) -> float:
    return json.loads((P39 / "floors.json").read_text())["floors"][rowset]["K0"]


def norms(rowset: str) -> dict[str, float]:
    """The per-matrix norms (84 MLP matrices) of the row set's seed-0 late fine-tune, as P3.9 run 1 recorded them."""
    return json.loads((P39 / "run1" / f"{rowset}_s0_last_kl.json").read_text())["random"]["norms"]


# ----------------------------------------------------------------------------- the built change's neurons


def usage(lab: Any, rows: Tensor, dev: Any) -> Tensor:
    """Mean |a| per MLP neuron [28, 6144] at W0 over rows, positions 1-255 (position 0, where a row's attention-sink
    activations sit, left out), a = down_proj's input = silu(gate) * up."""
    acc = torch.zeros(N_LAYERS, lab.model.config.intermediate_size, dtype=torch.float64, device=dev)

    def hook(layer: int) -> Any:
        def add(_m: Any, a: tuple[Tensor, ...], _o: Tensor) -> None:  # returns None: the module's output is kept
            acc[layer] += a[0][0, 1:].abs().double().sum(0)

        return add

    lab.use("W0")
    hs = [lyr.mlp.down_proj.register_forward_hook(hook(i)) for i, lyr in enumerate(lab.model.model.layers)]
    try:
        for r in rows:
            lab.model(r[None].to(dev), use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return (acc / (rows.shape[0] * (rows.shape[1] - 1))).cpu()


def interleave(per_layer: list[list[int]]) -> list[tuple[int, int]]:
    """Every layer's first neuron, then every layer's second, ...: a prefix of length k holds about k/28 per layer."""
    depth = max((len(p) for p in per_layer), default=0)
    return [(lyr, p[i]) for i in range(depth) for lyr, p in enumerate(per_layer) if i < len(p)]


def orders(u: dict[str, Tensor], rowset: str, live: list[Tensor], pool: float, seed: int) -> dict[str, list[tuple[int, int]]]:
    """Three orders of the neurons the model uses on `rowset`, each interleaved over layers. The used pool, per layer:
    the top `pool` share of the live gate neurons by mean |a| on the row set (`usage`). top: the pool by mean |a|,
    highest first; random: the pool in a seeded random order; specific: the pool by mean |a| on this row set over
    its mean on the other four, highest first (the neurons this text uses more than the library's other text)."""
    others = torch.stack([v for r, v in u.items() if r != rowset]).mean(0)
    per: dict[str, list[list[int]]] = {c: [] for c in CHOICES}
    for lyr, idx in enumerate(live):
        top = idx[u[rowset][lyr, idx].argsort(descending=True, stable=True)[: math.ceil(pool * len(idx))]]
        per["top"].append(top.tolist())
        gen = torch.Generator().manual_seed(seed * 1000 + lyr)
        per["random"].append(top[torch.randperm(len(top), generator=gen)].tolist())
        ratio = u[rowset][lyr, top] / others[lyr, top].clamp_min(1e-12)
        per["specific"].append(top[ratio.argsort(descending=True, stable=True)].tolist())
    return {c: interleave(v) for c, v in per.items()}


def turned_down(order: list[tuple[int, int]], x: float) -> list[tuple[int, int, float]]:
    """The (layer, neuron, factor) of size x: factor 0 on the first floor(x) neurons of `order` (switched off), then
    1 - frac(x) on the next (turned down partway), if frac(x) > 0 and the order has one."""
    assert 0 <= x <= len(order), (x, len(order))
    k, frac = math.floor(x), x - math.floor(x)
    out = [(lyr, c, 0.0) for lyr, c in order[:k]]
    return out + ([(*order[k], 1.0 - frac)] if frac > 0 and k < len(order) else [])


def built_change(W0: dict[str, Tensor], neurons: list[tuple[int, int, float]],
                 shapes: dict[str, tuple[int, ...]]) -> dict[str, Tensor]:
    """dW = (factor - 1)·W0 on the gate_proj rows of `neurons` (layer, index, factor), zero everywhere else."""
    by: dict[int, list[tuple[int, float]]] = {}
    for lyr, c, f in neurons:
        by.setdefault(lyr, []).append((c, f))
    part = {}
    for lyr, cfs in by.items():
        s = site_name(lyr, "gate")
        idx, fac = torch.tensor([c for c, _ in cfs]), torch.tensor([f for _, f in cfs], dtype=torch.float32)
        part[s] = torch.zeros(W0[s].shape)
        part[s][idx] = (fac - 1)[:, None] * W0[s][idx].float()
    return SR.with_zeros(part, shapes)


# ----------------------------------------------------------------------------- one random draw at three sizes


def random_draw(lab: Any, rowset: str, draw: int, rows: Tensor, shapes: dict[str, tuple[int, ...]],
                start: Callable[[str, tuple[float, float] | None, float], float], n_size: int, match: float,
                evals: int) -> None:
    """Gaussian draw `draw` at the row set's fine-tune norms, its scale b searched (on the first n_size rows, within
    `match`) so K00 lands at each of the three sizes, and read at each (`SR.read`, written as random/<rowset>_d<draw>_
    <level>). The split is linear in the change, so one split serves the three sizes (rescaled in place). `start(level,
    prev, excess)` gives the search's first scale; prev = (b, excess) of this draw's previous size. Reads on disk are
    skipped."""
    names = {lv: f"random/{rowset}_d{draw}_{lv}" for lv in LEVELS}
    todo = [lv for lv in LEVELS if not lab.get(names[lv])]
    if not todo:
        return
    fl, T = floor(rowset), sizes(rowset)
    full = SR.with_zeros(SR.gaussian(norms(rowset), shapes, draw), shapes)
    sp = lab.split(full, nulls=False, halves=False)
    SR.read_at_sizes(lab, {lv: names[lv] for lv in todo}, full, sp["X"], {lv: T[lv] - fl for lv in todo}, fl, rows,
                     start, n_size, match, evals,
                     lambda lv, b, search: {"decomp": FROZEN, "kind": "random", "rowset": rowset, "draw": draw,
                                            "level": lv, "b": b, "norms_from": f"{rowset}_s0_last", "target_k00": T[lv],
                                            "search": search, "stored_error": sp["stored_error"]})
