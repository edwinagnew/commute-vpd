"""Steering in the fair form the user set on 2026-11-09 (items 1-4 for cells P2.4 and P2.5): a pooled and a per-key vector,
a (layer, strength) chosen on fitting windows, norm-matched random vectors, cosines and off-target KL. First run as E135,
whose `experiments/steering_copy_lookup/run.py` stays as it ran; `pick`, `per_key_mean`, `per_window`, `random_like` and
`off_kl` are moved from there unchanged, and the vectors and reads are batched with the window keys passed in, so E140 can
key by the word before the trigger and read 9,600 fitting windows.

The target model only, fp32 (`steering.Net`). A vector at block l is the mean, over fitting windows, of (tuned - W0) block l's
input at the answer position `pos`; a per-key vector is that mean over the fitting windows of one key. Steering adds
strength x vector to W0's block-l input at `pos`, nothing else.

`steering_share` is the one steering share (the reading refactor, P2.79 move 5): the vector, the selection and the read
in one call, every setting that told the sources apart a required field of `Rule`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

from commute import share as SH
from commute import steering as ST
from commute.revival_kv import kl_rows

if TYPE_CHECKING:
    from commute.windows import Rows

LAYERS = tuple(range(ST.N_LAYERS))
BATCH = 64


def key_index(keys: Tensor, pool: Tensor) -> Tensor:
    """(N,) the index in `pool` (K,) of each window's key (N,); asserts every key is in the pool."""
    hits = keys[:, None] == pool[None]
    assert bool((hits.sum(1) == 1).all()), "a window's key is not in the pool"
    return hits.long().argmax(1)


def per_key_mean(diff: Tensor, idx: Tensor, n_keys: int) -> Tensor:
    """(n_keys, d): the mean of diff (N, d) over each key's windows (E135). An empty key would divide by 0: callers check."""
    out = torch.zeros(n_keys, diff.shape[1], device=diff.device, dtype=diff.dtype).index_add_(0, idx, diff)
    return out / torch.bincount(idx, minlength=n_keys).to(diff.dtype)[:, None]


def per_window(v: Tensor, idx: Tensor) -> Tensor:
    """A pooled vector (d,) as it is (it broadcasts over windows); a per-key vector (K, d) indexed by each window's key."""
    return v if v.ndim == 1 else v[idx]


def diffs(net: ST.Net, toks: Tensor, dW: ST.DW, pos: int, batch: int, dtype: torch.dtype) -> dict[int, Tensor]:
    """Per layer, (N, d) tuned minus W0 block-l input at `pos` on `toks`, `batch` windows per forward, cast to dtype."""
    out: dict[int, list[Tensor]] = {l: [] for l in LAYERS}
    for s in range(0, len(toks), batch):
        b = toks[s : s + batch]
        with net.at(dW):
            c1 = net.run(b, capture=True)[1]
        c0 = net.run(b, capture=True)[1]
        for l in LAYERS:
            out[l].append((c1[f"in{l}"][:, pos] - c0[f"in{l}"][:, pos]).to(dtype))
    return {l: torch.cat(x) for l, x in out.items()}


def means(d: dict[int, Tensor], idx: Tensor | None, n_keys: int) -> dict[int, Tensor]:
    """Per layer, the mean of `diffs` (N, d) over every window (idx None: pooled, (d,)) or over each key's windows
    (per_key_mean, (n_keys, d)), cast to fp32: the one pooled and per-key mean (`vectors`, `fit_vectors`)."""
    if idx is None:
        return {l: x.mean(0).float() for l, x in d.items()}
    return {l: per_key_mean(x, idx.to(x.device), n_keys).float() for l, x in d.items()}


def vectors(net: ST.Net, toks: Tensor, idx: Tensor, n_keys: int, dW: ST.DW, pos: int) -> dict[str, dict[int, Tensor]]:
    """Per layer, the pooled (d,) and per-key (K, d) `means` of tuned minus W0 block-l input at `pos` over `toks`
    (batch 64, fp64), from one pass of forwards."""
    d = diffs(net, toks, dW, pos, BATCH, torch.float64)
    assert bool((torch.bincount(idx, minlength=n_keys) > 0).all()), "a key has no fitting window"
    return {"pooled": means(d, None, n_keys), "perkey": means(d, idx, n_keys)}


def steer(v: Tensor, layer: int, strength: float, idx: Tensor, pos: int) -> ST.Edit:
    return {f"in{layer}": ST.add_at(pos, strength * per_window(v, idx.to(v.device)))}


@torch.no_grad()
def logits_steered(net: ST.Net, toks: Tensor, idx: Tensor, pos: int, v: Tensor, layer: int, strength: float) -> Tensor:
    """(N, V) fp32 logits at `pos`, on the net's device, of W0 steered by v at (layer, strength), batched with each batch's
    own keys."""
    return torch.cat([net.run(toks[s : s + BATCH], steer(v, layer, strength, idx[s : s + BATCH], pos))[0][:, pos]
                      for s in range(0, len(toks), BATCH)])  # fmt: skip


def pick(rises: dict[tuple[int, float], float]) -> tuple[int, float] | None:
    """The filed selection: the largest rise, ties to the lower layer, then the lower strength. A nan among the rises
    leaves the setting undefined (None), so it cannot be silently ordered (E135)."""
    if any(math.isnan(v) for v in rises.values()):
        return None
    return max(rises, key=lambda ls: (rises[ls], -ls[0], -ls[1]))


def random_like(v: Tensor, gen: torch.Generator) -> Tensor:
    """A Gaussian direction with v's shape, scaled to v's norm (row by row for a per-key vector) (E135)."""
    g = torch.randn(v.shape, generator=gen).to(v.device, v.dtype)
    return g / g.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)


def cosine_matrix(vs: list[Tensor]) -> Tensor:
    """(n, n) float64 CPU cosines between pooled vectors (d,)."""
    m = torch.nn.functional.normalize(torch.stack([v.double().flatten() for v in vs]), dim=1)
    return (m @ m.T).cpu()


@torch.no_grad()
def off_kl(net: ST.Net, toks: Tensor, pos: int, idx: Tensor | None = None, steering: tuple[Tensor, int, float] | None = None,
           dW: ST.DW | None = None) -> float:  # fmt: skip
    """Mean KL(W0 || steered or state) over the positions other than `pos` (E135's `off_kl`, batched). Only positions after
    `pos` can move, since steering edits `pos` alone."""
    tot, n = 0.0, 0
    for s in range(0, len(toks), BATCH):
        b = toks[s : s + BATCH]
        edit = None if steering is None or idx is None else steer(steering[0], steering[1], steering[2], idx[s : s + BATCH], pos)
        l0 = torch.log_softmax(net.run(b)[0].double(), -1)
        with net.at(dW):
            l1 = torch.log_softmax(net.run(b, edit)[0].double(), -1)
        kl = kl_rows(l0, l1)
        keep = torch.ones(kl.shape[1], dtype=torch.bool, device=kl.device)
        keep[pos] = False
        tot, n = tot + float(kl[:, keep].sum()), n + int(kl[:, keep].numel())
    return tot / n


# ----------------------------------------------------------------------------- the steering share, one function (P2.79 move 5)


@dataclass(frozen=True)
class Rule:
    """How a source chose and read its steering setting; every field is required, since the sources differ in each
    (plan reports/plans/2026-11-25_plan_reading-refactor.md, finding 4):
      E135 (steering_copy_lookup)  fit = select = E62's even windows, read = the odd ones; key_pos half - 1; floor None;
                                   batch 100 (one forward of each half); mean float32
      E143 (holes_fill)            fit = the task's training windows, select = the first n_select of them, read = E136's
                                   rows; key_pos the read position; floor 1; batch 64; mean float64
      E211 / E272 (realistic_reads.Reader.steering)  as E143, pooled only, the unit the task is scored in (E140's
                                   two-label contrast for a label task)"""

    form: Literal["pooled", "perkey"]
    key_pos: int | None  # per-key: the column whose token keys each window (None for pooled)
    strengths: tuple[float, ...]
    floor: float | None  # the share is nan when the gain is under floor; None: only a zero gain (E135, `steering.share`)
    batch: int  # windows per forward
    mean: torch.dtype  # the dtype the vector's mean is taken in


def divide(num: float, gain: float, floor: float | None) -> float:
    """num / gain under `share.Floor`, one-sided (floor None: only a zero gain is not read)."""
    return num / gain if SH.Floor(-math.inf if floor is None else floor, two_sided=False).passes(gain) else math.nan


@torch.no_grad()
def unit_rows(net: ST.Net, rows: Rows, unit: ST.Readout, batch: int, dW: ST.DW | None = None,
              how: tuple[Tensor, int, float, Tensor] | None = None) -> Tensor:  # fmt: skip
    """(N,) float64: `unit` at rows.pos of W0 (or W0 + dW), steered by how = (vector, layer, strength, key index per row)."""
    tgt, out = rows.tgt, []
    assert tgt is not None, "steering reads rows with a target"
    for s in range(0, len(rows.toks), batch):
        edit = None if how is None else steer(how[0], how[1], how[2], how[3][s : s + batch], rows.pos)
        with net.at(dW):
            lg = net.run(rows.toks[s : s + batch], edit)[0]
        out.append(unit(lg, rows.pos, tgt[s : s + batch]))
    return torch.cat(out)


def target_gain(net: ST.Net, rows: Rows, dW: ST.DW, unit: ST.Readout, batch: int) -> float:
    """The fine-tune's own gain on `rows`, fp32 target model: mean(unit at W0 + dW - unit at W0) (E135's and E143's)."""
    return float((unit_rows(net, rows, unit, batch, dW) - unit_rows(net, rows, unit, batch)).mean())


def key_idx(rows: Rows, rule: Rule, pool: Tensor | None) -> Tensor:
    """(N,) each row's key in `pool` (zeros for a pooled rule)."""
    return torch.zeros(len(rows.toks), dtype=torch.long) if pool is None else key_index(rows.toks[:, rule.key_pos], pool)


def fit_vectors(net: ST.Net, fit: Rows, dW: ST.DW, rule: Rule, pool: Tensor | None) -> dict[int, Tensor]:
    """Per layer, the steering vector from `fit`: the mean of tuned minus W0 block input at fit.pos, pooled (d,) or per
    key over `pool` (K, d), taken in rule.mean and cast to fp32."""
    d = diffs(net, fit.toks, dW, fit.pos, rule.batch, rule.mean)
    return means(d, None, 0) if pool is None else means(d, key_idx(fit, rule, pool), len(pool))


def selection(net: ST.Net, fit: Rows, select: Rows, dW: ST.DW, rule: Rule, unit: ST.Readout, pool: Tensor | None) -> tuple[dict[int, Tensor], dict[str, Any]]:
    """The vector per layer from `fit` (pooled, or per key over `pool`) and the (layer, strength) with the largest mean
    rise of `unit` over W0 on `select` (`pick`): (vectors, {"grid", "setting"}); setting None when a rise is nan."""
    assert (pool is None) == (rule.form == "pooled") == (rule.key_pos is None), "a per-key rule needs its pool and key column"
    v, idx = fit_vectors(net, fit, dW, rule, pool), key_idx(select, rule, pool)
    sel0 = unit_rows(net, select, unit, rule.batch)
    grid = {(l, s): float((unit_rows(net, select, unit, rule.batch, how=(v[l], l, s, idx)) - sel0).mean())
            for l in LAYERS for s in rule.strengths}  # fmt: skip
    return v, {"grid": {f"{l},{s}": x for (l, s), x in grid.items()}, "setting": pick(grid)}


def steering_share(net: ST.Net, fit: Rows, select: Rows, read: Rows, dW: ST.DW, rule: Rule, unit: ST.Readout, gain: float,
                   pool: Tensor | None) -> dict[str, Any]:  # fmt: skip
    """The fair steering share of one update: `selection` on `fit` and `select`, then the chosen setting's mean rise
    over W0 on `read` divided by `gain` (the caller's: E135 and E143 read `target_gain`, E211 and E272 the share's T
    cells). All rows share one read position."""
    assert fit.pos == select.pos == read.pos
    v, rec = selection(net, fit, select, dW, rule, unit, pool)
    if rec["setting"] is None:
        return rec | {"rise": math.nan, "share": math.nan}
    l, s = rec["setting"]
    idx = key_idx(read, rule, pool)
    rise = float((unit_rows(net, read, unit, rule.batch, how=(v[l], l, s, idx)) - unit_rows(net, read, unit, rule.batch)).mean())
    return rec | {"rise": rise, "share": divide(rise, gain, rule.floor)}

