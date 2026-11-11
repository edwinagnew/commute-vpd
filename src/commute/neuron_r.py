"""E98's neuron analogue of VPD's r on the 67M (experiments/neuron_r; plan reports/plans/2026-11-05_plan_e98-neuron-r.md).
Target model only, fp32, no decomposition read. The readings are below; `experiments/neuron_r/run.py` loads states and
windows and calls them; the filed rules are in its results/predictions.json.

The 67M's block l (param_decomp LlamaSimpleMLP, `artifacts/vpd/t-9d2b8f02`, read 2026-11-05) computes
x + attn(rms_1(x)), then x + down_proj(gelu(c_fc(rms_2(x)))); no biases; the 24 decomposed matrices are the four
attention projections, c_fc and down_proj per block. A neuron is one of c_fc's 3072 outputs after the GELU (one input of
down_proj): 4 x 3072 = 12288, flat index l x 3072 + i as `neurons.py`. `forward` hooks two places per block: the input
of `block.mlp` ("the MLP input", rms_2's output) and the input of `down_proj` ("the firing"), every position.

Cells, per window, log-odds of the target at the answer position (`steering.logodds_at`, the route split's unit). Every
cell but T1 runs on W0's attention weights. O = down_proj's weights (O0 W0's, O1 the state's), A = the firing down_proj
sees at every block and position (A0 W0's own, A1 the state's, each captured from that model's forward on the same
window), Ah = W0's c_fc and GELU on the state's MLP input (x1, captured from the state's forward).
  r_half  L(O0, Ah) - T0   both MLP matrices W0's, only the MLP input changed: upstream re-firing (the headline)
  r_n     L(O0, A1) - T0   W0's output weights on the state's firing: E52/E56's full-width patch
  d_n     L(O1, A0) - T0   the state's output weights on W0's firing
  i_n     L(O1, A1) - L(O0, A1) - L(O1, A0) + T0
  e_n     T1 - L(O1, A1)   the rest: the attention matrices' change, the MLP outputs held at the state's
r_n + d_n + i_n + e_n = T1 - T0 = G by telescoping; a share is the mean numerator over the mean G (as VPD's r,
`local_faith.py` route split), defined when |G| reaches the gain floor. r_n - r_half is what c_fc's change adds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from commute import elicit as EL
from commute import steering as ST
from commute.identity import log_odds
from commute.steering import D_MLP, DW, N_LAYERS, Net

N_NEURONS = N_LAYERS * D_MLP
SHARES = ("r_half", "r_n", "d_n", "i_n", "e_n")
Acts = dict[int, Tensor]  # block -> (B, L, width) as the forward used it


@dataclass(frozen=True)
class Reads:
    """One task's read windows: the tokens the model sees, the target per window, the answer position."""

    name: str
    toks: Tensor  # (N, L)
    tgt: Tensor  # (N,)
    half: int

    def batches(self, size: int) -> Iterator[tuple[Tensor, Tensor]]:
        for s in range(0, len(self.toks), size):
            yield self.toks[s : s + size], self.tgt[s : s + size]


def task_reads(t: ST.Task) -> Reads:
    """A `steering.Task` (key -> value table, or E53's next-token `NextTask`) as Reads on all its held-out windows."""
    return Reads(t.name, t.toks, t.tgt(t.toks).cpu(), t.half)


def out_only(net: Net, dW: DW) -> DW:
    """The state's down_proj change alone, the other 18 matrices at 0 (`Net.at` needs all 24)."""
    return {k: dW[k] if k.endswith("mlp.down_proj") else torch.zeros_like(w) for k, w in net.W0.items()}


def hooked(mlps: Sequence[Any], call: Callable[[], Any], x_in: Acts | None = None, a_in: Acts | None = None) -> tuple[Any, Acts, Acts]:
    """`call()` with a pre-hook on each MLP module (`.down_proj` inside it): block l's MLP input replaced by x_in[l], its
    firing (down_proj's input) by a_in[l], when given. Returns call's output and each block's MLP input and firing as
    used. Any model whose MLPs have a `down_proj` (the 67M's LlamaSimpleMLP, Qwen3's Qwen3MLP); generalised for E102."""
    xs: Acts = {}
    acts: Acts = {}

    def at(l: int, given: Acts | None, seen: Acts) -> Any:
        def f(_m: Any, a: tuple[Tensor, ...]) -> tuple[Tensor]:
            seen[l] = a[0] if given is None else given[l]
            return (seen[l],)

        return f

    handles = []
    for l, mlp in enumerate(mlps):
        handles.append(mlp.register_forward_pre_hook(at(l, x_in, xs)))
        handles.append(mlp.down_proj.register_forward_pre_hook(at(l, a_in, acts)))
    try:
        out = call()
    finally:
        for h in handles:
            h.remove()
    return out, xs, acts


@torch.no_grad()
def forward(net: Net, toks: Tensor, x_in: Acts | None = None, a_in: Acts | None = None) -> tuple[Tensor, Acts, Acts]:
    """One forward of the 67M as it stands (`Net.at` sets the weights), through `hooked`. Returns the logits and each
    block's MLP input and firing as used."""
    out, xs, acts = hooked([b.mlp for b in net.net.h], lambda: net.net(toks.to(net.device)), x_in, a_in)
    return (out[0] if isinstance(out, tuple) else out).float(), xs, acts


def batch_cells(net: Net, toks: Tensor, tgt: Tensor, half: int, dW: DW, unit: ST.Readout = ST.logodds_at) -> dict[str, Tensor]:
    """Per window of one batch, every cell (T0, T1, L00 = W0 with its own firing put back, L01, L10, L11, Lh) in `unit`
    (log-odds of the target unless given; E158 passes the two-label contrast), plus log P of T0, T1 and L01 (E56's unit,
    for the reproduction) — float64 on CPU."""
    lo = lambda lg: unit(lg, half, tgt)
    lp = lambda lg: ST.logp_at(lg, half, tgt)
    lg, _, a0 = forward(net, toks)
    out = {"T0": lo(lg), "lp_T0": lp(lg)}
    with net.at(dW):
        lg, x1, a1 = forward(net, toks)
    out |= {"T1": lo(lg), "lp_T1": lp(lg)}
    out["L00"] = lo(forward(net, toks, a_in=a0)[0])
    lg = forward(net, toks, a_in=a1)[0]
    out |= {"L01": lo(lg), "lp_L01": lp(lg)}
    out["Lh"] = lo(forward(net, toks, x_in=x1)[0])
    with net.at(out_only(net, dW)):
        out["L10"] = lo(forward(net, toks, a_in=a0)[0])
        out["L11"] = lo(forward(net, toks, a_in=a1)[0])
    return out


def state_cells(net: Net, r: Reads, dW: DW, batch: int, unit: ST.Readout = ST.logodds_at) -> dict[str, Tensor]:
    parts = [batch_cells(net, toks, tgt, r.half, dW, unit) for toks, tgt in r.batches(batch)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def max_deviation(c: dict[str, Tensor]) -> float:
    """The largest |cell - T0| over windows and cells: W0's reading, 0 when every patch puts back what was there."""
    return max(float((c[k] - c["T0"]).abs().max()) for k in ("T1", "L00", "L01", "L10", "L11", "Lh"))


def shares(c: dict[str, Tensor], floor: float) -> dict[str, Any]:
    """G, each cell's mean numerator in log-odds, its share of G (nan when |G| < floor), mean P(target) in the state,
    the identity reading (max |L00 - T0|) and the telescoping residual (r_n + d_n + i_n + e_n - G)."""
    m = lambda x: float(x.mean())
    T0, T1 = c["T0"], c["T1"]
    num = {"r_half": m(c["Lh"] - T0), "r_n": m(c["L01"] - T0), "d_n": m(c["L10"] - T0),
           "i_n": m(c["L11"] - c["L01"] - c["L10"] + T0), "e_n": m(T1 - c["L11"])}
    G = m(T1 - T0)
    readable = abs(G) >= floor
    return {"G": G, "num": num, **{k: v / G if readable else math.nan for k, v in num.items()}, "readable": readable,
            "p_target": float(c["lp_T1"].exp().mean()), "identity": float((c["L00"] - T0).abs().max()),
            "telescope": sum(num[k] for k in ("r_n", "d_n", "i_n", "e_n")) - G, "n_windows": len(T0)}


# ----------------------------------------------------------------------------- the carrying set (P3)


def firing_grads(mlps: Sequence[Any], loss: Callable[[], Tensor]) -> tuple[Acts, Acts]:
    """d loss() / d(firing) at every block and position, the total derivative: a zero leaf added at each down_proj input
    (as `ctxpatch.batch_scores`), so paths through later neurons count. Also returns the firing itself (detached).
    Generalised from w0_grads for E102; `loss` runs the model and returns a scalar."""
    leaves: Acts = {}
    seen: Acts = {}

    def hook(l: int) -> Any:
        def f(_m: Any, a: tuple[Tensor, ...]) -> tuple[Tensor]:
            leaves[l] = torch.zeros_like(a[0]).requires_grad_(True)
            seen[l] = a[0].detach()
            return (a[0] + leaves[l],)

        return f

    handles = [m.down_proj.register_forward_pre_hook(hook(l)) for l, m in enumerate(mlps)]
    try:
        with torch.enable_grad():
            grads = torch.autograd.grad(loss(), [leaves[l] for l in range(len(mlps))])
    finally:
        for h in handles:
            h.remove()
    return dict(enumerate(grads)), seen


def w0_grads(net: Net, toks: Tensor, tgt: Tensor, half: int) -> Acts:
    """d(sum over windows of W0's log-odds at `half`) / d(firing), every block and position (`firing_grads`)."""

    def loss() -> Tensor:
        out = net.net(toks.to(net.device))
        lg = (out[0] if isinstance(out, tuple) else out).float()
        return log_odds(lg[:, half], tgt.to(net.device)).sum()

    return firing_grads([b.mlp for b in net.net.h], loss)[0]


def half_acts(net: Net, toks: Tensor, dW: DW) -> tuple[Tensor, Acts, Acts]:
    """W0's logits, W0's firing A0 and Ah (W0's c_fc and GELU on the state's MLP input) on one batch."""
    lg, _, a0 = forward(net, toks)
    with net.at(dW):
        x1 = forward(net, toks)[1]
    return lg, a0, forward(net, toks, x_in=x1)[2]


def ranking(net: Net, r: Reads, dW: DW, batch: int) -> Tensor:
    """(12288,) float64 per neuron: sum over windows and positions of W0's gradient x (Ah - A0), the first-order estimate
    of its part in the r_half cell's numerator (summed, not averaged: only the order is used)."""
    score = torch.zeros(N_NEURONS, dtype=torch.float64)
    for toks, tgt in r.batches(batch):
        _, a0, ah = half_acts(net, toks, dW)
        g = w0_grads(net, toks, tgt, r.half)
        score += torch.cat([(g[l] * (ah[l] - a0[l])).sum((0, 1)).double().cpu() for l in range(N_LAYERS)])
    return score


def top_masks(order: Tensor, k: int, n_layers: int = N_LAYERS, width: int = D_MLP) -> dict[int, Tensor]:
    """Per block, (width,) bool: the first k neurons of `order` (flat indices l x width + i)."""
    flat = torch.zeros(n_layers * width, dtype=torch.bool)
    flat[order[:k]] = True
    return {l: flat[l * width : (l + 1) * width] for l in range(n_layers)}


def grid_numerators(net: Net, r: Reads, dW: DW, order: Tensor, grid: Sequence[int], batch: int) -> dict[int, float]:
    """Per k: the mean over windows of L(O0, A_k) - T0, A_k = Ah on the first k neurons of `order` and A0 on the rest,
    every block and position. At k = 12288 it is the r_half numerator."""
    masks = {k: top_masks(order, k) for k in grid}
    sums = dict.fromkeys(grid, 0.0)
    for toks, tgt in r.batches(batch):
        lg, a0, ah = half_acts(net, toks, dW)
        T0 = ST.logodds_at(lg, r.half, tgt)
        for k in grid:
            a_in = {l: torch.where(masks[k][l].to(a0[l].device), ah[l], a0[l]) for l in range(N_LAYERS)}
            sums[k] += float((ST.logodds_at(forward(net, toks, a_in=a_in)[0], r.half, tgt) - T0).sum())
    return {k: s / len(r.toks) for k, s in sums.items()}


def k_half(nums: dict[int, float], full: float, share: float, carry_floor: float) -> int | None:
    """The first grid k whose numerator reaches half of `full` (the r_half numerator); None ('no carrying set') when the
    r_half share is not above `carry_floor` or its numerator is not positive (then half of it is no target). With
    full > 0 the last k, equal to full (checked by the caller), always qualifies."""
    if not (share > carry_floor and full > 0):
        return None
    return next(k for k in sorted(nums) if nums[k] >= full / 2)


# ----------------------------------------------------------------------------- the filed rules


def scored(rec: dict[str, Any], p_learn: float | None) -> bool:
    """A state enters a rule when its gain reaches the floor and, for a key -> value task (p_learn not None), its mean
    held-out P(target) reaches p_learn (`keydep.P_LEARN`, E12's learned rule)."""
    return bool(rec["readable"]) and (p_learn is None or rec["p_target"] >= p_learn)


def row_sides(recs: dict[str, dict[str, Any]], row: dict[str, Any], key: str, filtered: bool = True) -> tuple[list[float], list[float]]:
    """A row's reuse and new values of `key`, on its scored states (`scored`, stored per state as rec['scored']), or on
    every state whose gain reaches the floor when not `filtered`."""
    keep = lambda n: bool(recs[n]["scored"] if filtered else recs[n]["readable"])
    return [recs[n][key] for n in row["reuse"] if keep(n)], [recs[n][key] for n in row["new"] if keep(n)]


def row_verdict(recs: dict[str, dict[str, Any]], row: dict[str, Any], key: str, S: dict[str, Any], filtered: bool = True) -> dict[str, Any]:
    """E51/E52's rule (`elicit.separation`, min gap) on one row: 'a_higher' = the reuse side higher."""
    a, b = row_sides(recs, row, key, filtered)
    return EL.separation(list(a), list(b), S["min_gap"])


def majority_fails(verdicts: dict[str, dict[str, Any]], min_rows: int) -> dict[str, Any]:
    """P1: the reading separates reuse-higher on fewer than half of the readable rows; 'unreadable' under min_rows."""
    readable = [n for n, v in verdicts.items() if v["verdict"] != "unreadable"]
    sep = [n for n in readable if verdicts[n]["verdict"] == "a_higher"]
    status = "unreadable" if len(readable) < min_rows else "pass" if 2 * len(sep) < len(readable) else "fail"
    return {"status": status, "readable_rows": readable, "separating_rows": sep, "rows": verdicts}


def outside_share(recs: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]], S: dict[str, Any]) -> dict[str, Any]:
    """P2: e_n separates reuse-higher on every readable row, with at least min_rows readable. A row with a scored state
    whose |i_n| exceeds S['unstable_i'] reads 'cells_unstable', no verdict."""
    out: dict[str, Any] = {}
    for name, row in rows.items():
        a, b = row_sides(recs, row, "i_n")
        if any(abs(x) > S["unstable_i"] for x in a + b):
            out[name] = {"verdict": "cells_unstable", "i_n": a + b}
        else:
            out[name] = row_verdict(recs, row, "e_n", S)
    readable = [n for n, v in out.items() if v["verdict"] not in ("unreadable", "cells_unstable")]
    ok = all(out[n]["verdict"] == "a_higher" for n in readable)
    status = "unreadable" if len(readable) < S["min_rows"] else "pass" if ok else "fail"
    return {"status": status, "readable_rows": readable, "rows": out}


def carrying_rule(ks: dict[str, int | None], low: Sequence[str], low_max: int, other_min: int) -> dict[str, Any]:
    """P3: every defined k_half of a `low` state (lowercase) is <= low_max and every other defined one is > other_min.
    A violation fails it; otherwise it is 'unreadable' unless both groups have a defined state."""
    lows = {n: k for n, k in ks.items() if n in low and k is not None}
    others = {n: k for n, k in ks.items() if n not in low and k is not None}
    bad = [n for n, k in lows.items() if k > low_max] + [n for n, k in others.items() if k <= other_min]
    status = "fail" if bad else "unreadable" if not lows or not others else "pass"
    return {"status": status, "violations": bad, "low": lows, "other": others,
            "no_carrying_set": sorted(n for n, k in ks.items() if k is None)}
