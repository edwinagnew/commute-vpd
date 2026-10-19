"""Recruitment by the real update (`experiments/recruit_test`).

With the components left at their released values and Gamma frozen, a change to the target weights (a fine-tune's
delta-W put into Delta) moves the activations Gamma reads, so some subcomponents switch on at the trigger. This module
is the pure bookkeeping around that: the switched-on set, its comparison across updates (Jaccard), hybrid gate patterns
for the recovery measure, random sets of matched size, and the contrast of W0's gates at positions that precede a given
token (the model's natural machinery for that token). Gate patterns are lists of per-batch dicts, as
`local_faith.route_cells` returns them: `gates[b][key]` is (B, T, C) on CPU.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

Gates = list[dict[str, Tensor]]
Sel = dict[str, Tensor]  # matrix key -> 1-D long tensor of component indices

_LAYER_RE = re.compile(r"^h\.(\d+)\.")


def layer_or_none(key: str) -> int | None:
    m = _LAYER_RE.match(key)
    return int(m.group(1)) if m else None


def cat_gates(gates: Gates) -> dict[str, Tensor]:
    return {k: torch.cat([g[k] for g in gates]) for k in gates[0]}


def trigger_dg(gates_state: Gates, gates_base: Gates, half: int, alive: dict[str, Tensor]) -> dict[str, Tensor]:
    """Mean over windows of g_state - g_base at the trigger position, per subcomponent (float32, CPU). Dead
    subcomponents (not in `alive[k]`, a bool mask) are set to 0 so they never count as switched on or off."""
    g1, g0 = cat_gates(gates_state), cat_gates(gates_base)
    out = {}
    for k in g0:
        d = (g1[k][:, half].float() - g0[k][:, half].float()).mean(0).cpu()
        out[k] = torch.where(alive[k].cpu().bool(), d, torch.zeros_like(d))
    return out


def switched(dg: dict[str, Tensor], thr: float, on: bool = True) -> Sel:
    """Indices with dg > thr (switched on) or dg < -thr (switched off)."""
    return {k: ((v > thr) if on else (v < -thr)).nonzero().flatten() for k, v in dg.items()}


def size(sel: Sel) -> int:
    return int(sum(int(v.numel()) for v in sel.values()))


def members(sel: Sel) -> set[tuple[str, int]]:
    return {(k, int(c)) for k, v in sel.items() for c in v.tolist()}


def jaccard(a: Sel, b: Sel) -> float:
    """|A ∩ B| / |A ∪ B|; nan when both are empty."""
    sa, sb = members(a), members(b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else float("nan")


def by_layer(sel: Sel) -> dict[str, int]:
    """Member counts per layer (keys without an `h.<n>.` prefix are counted under 'other')."""
    out: dict[str, int] = {}
    for k, v in sel.items():
        lay = layer_or_none(k)
        name = str(lay) if lay is not None else "other"
        out[name] = out.get(name, 0) + int(v.numel())
    return dict(sorted(out.items()))


def restrict(sel: Sel, min_layer: int) -> Sel:
    """Only the matrices in layers >= min_layer (keys without a layer are dropped)."""
    out = {}
    for k, v in sel.items():
        lay = layer_or_none(k)
        if lay is not None and lay >= min_layer:
            out[k] = v
    return out


def hybrid(gates_base: Gates, gates_state: Gates, sel: Sel) -> Gates:
    """The base gate pattern, except that the subcomponents in `sel` take the state's gates at every position."""
    out: Gates = []
    for g0, g1 in zip(gates_base, gates_state, strict=True):
        h = {}
        for k, v in g0.items():
            v = v.clone()
            if k in sel and sel[k].numel():
                idx = sel[k].to(v.device)
                v[..., idx] = g1[k][..., idx].to(v.dtype)
            h[k] = v
        out.append(h)
    return out


def random_like(sel: Sel, alive: dict[str, Tensor], seed: int) -> Sel:
    """A set with the same size per matrix as `sel`, drawn from the alive subcomponents not in `sel`."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, idx in sel.items():
        pool = alive[k].cpu().bool().clone()
        pool[idx.cpu()] = False
        cand = pool.nonzero().flatten()
        n = min(int(idx.numel()), int(cand.numel()))
        out[k] = cand[torch.randperm(cand.numel(), generator=gen)[:n]] if n else torch.zeros(0, dtype=torch.long)
    return out


def precede_mask(rows: Tensor, token: int) -> Tensor:
    """(B, T) bool: True at the positions whose NEXT token is `token` (the last position is never True)."""
    m = torch.zeros_like(rows, dtype=torch.bool)
    m[:, :-1] = rows[:, 1:] == token
    return m


class ContrastAcc:
    """Accumulates, per subcomponent, the mean gate at masked positions minus the mean gate over all positions."""

    def __init__(self) -> None:
        self.sum_on: dict[str, Tensor] = {}
        self.sum_all: dict[str, Tensor] = {}
        self.n_on = 0
        self.n_all = 0

    def add(self, g: dict[str, Tensor], mask: Tensor) -> None:
        mask = mask.to(next(iter(g.values())).device)
        for k, v in g.items():
            v = v.float()
            on = v[mask].sum(0).double().cpu()
            al = v.sum(dim=(0, 1)).double().cpu()
            self.sum_on[k] = self.sum_on.get(k, 0) + on
            self.sum_all[k] = self.sum_all.get(k, 0) + al
        self.n_on += int(mask.sum())
        self.n_all += int(mask.numel())

    def result(self, alive: dict[str, Tensor]) -> dict[str, Tensor]:
        out = {}
        for k in self.sum_all:
            mean_on = self.sum_on[k] / max(self.n_on, 1)
            mean_all = self.sum_all[k] / max(self.n_all, 1)
            d = (mean_on - mean_all).float()
            out[k] = torch.where(alive[k].cpu().bool(), d, torch.zeros_like(d))
        return out
