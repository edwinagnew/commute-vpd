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


# ----------------------------------------------------------------------------- trigger-only gate patterns (E18)
# Added 2026-10-21 for `experiments/regating_leak` (reports/2026-10-21_plan_regating-leak.md). Each pattern is W0's gates
# everywhere except the trigger position, where it writes per-window values `trig[b][key]` (B, C).

TrigVals = list[dict[str, Tensor]]


def at_trigger(gates: Gates, half: int) -> TrigVals:
    """Each window's own gates at the trigger position."""
    return [{k: v[:, half] for k, v in g.items()} for g in gates]


def added(gates_base: Gates, half: int, shift: dict[str, Tensor]) -> TrigVals:
    """Each window's base gates at the trigger plus one shared (C,) shift per matrix, not clipped (values may leave
    [0, 1]; E18 amendment 1's arm that isolates the clip)."""
    return [{k: v[:, half].float() + shift[k].float() for k, v in g.items()} for g in gates_base]


def shifted(gates_base: Gates, half: int, shift: dict[str, Tensor]) -> TrigVals:
    """`added`, clipped to [0, 1]."""
    return [{k: v.clamp(0.0, 1.0) for k, v in t.items()} for t in added(gates_base, half, shift)]


def constant(gates_base: Gates, half: int, vec: dict[str, Tensor]) -> TrigVals:
    """The same (C,) vector per matrix at the trigger of every window: no per-window term at all."""
    return [{k: vec[k].float().expand(v.shape[0], -1) for k, v in g.items()} for g in gates_base]


def with_trigger(gates_base: Gates, half: int, trig: TrigVals) -> Gates:
    """The base gate pattern with `trig` written at the trigger position (every other position untouched)."""
    out: Gates = []
    for g0, t in zip(gates_base, trig, strict=True):
        h = {}
        for k, v in g0.items():
            v = v.clone()
            v[:, half] = t[k].to(v.device, v.dtype)
            h[k] = v
        out.append(h)
    return out


def mean_change(mean_state: dict[str, Tensor], mean_base: dict[str, Tensor], alive: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per subcomponent, mean trigger gate under the state minus under W0 (float32, CPU); dead subcomponents set to 0."""
    return {k: torch.where(alive[k].cpu().bool(), (v.float() - mean_base[k].float()).cpu(), torch.zeros(v.shape))
            for k, v in mean_state.items()}


def moved(shift: dict[str, Tensor], sel: Sel, dest: Sel) -> dict[str, Tensor]:
    """A shift that is zero except at `dest`, whose members take the values `shift` has at `sel`'s members, matrix by
    matrix in order: as much switching-on as `sel` gets, put on other subcomponents (the random null)."""
    out = {}
    for k, v in shift.items():
        m = torch.zeros_like(v)
        d = dest.get(k)
        if d is not None and d.numel():
            m[d] = v[sel[k][: d.numel()]]
        out[k] = m
    return out


def shuffled(shift: dict[str, Tensor], alive: dict[str, Tensor], seed: int) -> dict[str, Tensor]:
    """The shift's values permuted among each matrix's alive subcomponents (every entry, one torch generator per draw,
    matrices in dict order); dead subcomponents stay 0. Keeps each matrix's set of values, loses which subcomponent
    gets which (E18 amendment 2's null for the whole shift)."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in shift.items():
        idx = alive[k].cpu().bool().nonzero().flatten()
        m = torch.zeros_like(v)
        m[idx] = v[idx][torch.randperm(idx.numel(), generator=gen)]
        out[k] = m
    return out


def in_range(shift: dict[str, Tensor], alive: dict[str, Tensor], lo: float, hi: float) -> Sel:
    """Alive subcomponents with lo < shift <= hi."""
    return {k: (alive[k].cpu().bool() & (v > lo) & (v <= hi)).nonzero().flatten() for k, v in shift.items()}


# ----------------------------------------------------------------------------- amendment 3 (E18, 2026-10-21)
# Arms that the leading hypothesis H can fail (reports/2026-10-21_plan_regating-leak.md -> Amendment 3): the updated
# model's own trigger values with the fixed change removed, another window's values, the change's two signs.


def plus(trig: TrigVals, shift: dict[str, Tensor], clip: bool) -> TrigVals:
    """Trigger values plus one (C,) shift per matrix, clipped to [0, 1] if `clip` (`added` and `shifted` for any
    trigger values, e.g. the state's own: plus(at_trigger(g1), -s) is the residual g1(x) - s)."""
    out = [{k: v.float() + shift[k].float() for k, v in t.items()} for t in trig]
    return [{k: v.clamp(0.0, 1.0) for k, v in t.items()} for t in out] if clip else out


def derangement(n: int, seed: int) -> Tensor:
    """A random permutation of range(n) with no fixed point (one cycle through a random order), n >= 2."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    perm = torch.empty(n, dtype=torch.long)
    perm[order] = order.roll(-1)
    return perm


def permuted(trig: TrigVals, perm: Tensor) -> TrigVals:
    """Window i gets window perm[i]'s trigger values; the batch layout is kept."""
    sizes = [next(iter(t.values())).shape[0] for t in trig]
    cat = {k: torch.cat([t[k] for t in trig])[perm] for k in trig[0]}
    return [dict(zip(cat, vals, strict=True)) for vals in zip(*(v.split(sizes) for v in cat.values()), strict=True)]


def signed(shift: dict[str, Tensor], positive: bool) -> dict[str, Tensor]:
    """The shift's positive (turned-up) or negative (turned-down) entries, 0 elsewhere."""
    return {k: v.clamp(min=0.0) if positive else v.clamp(max=0.0) for k, v in shift.items()}


def on_pool(shift: dict[str, Tensor], pool: dict[str, Tensor]) -> dict[str, Tensor]:
    """The shift on the `pool` (bool per matrix) only, 0 elsewhere."""
    return {k: torch.where(pool[k].cpu().bool(), v, torch.zeros_like(v)) for k, v in shift.items()}
