"""Per-window reads of a two-label task at the answer position, in three units, for the re-gating share and its neuron
version (E140, cell P2.4; the user's items 5 and 7 of 2026-11-09).

Units, from the logits at `pos`, with y = 1 for label A and 0 for B:
  lo        log-odds of the window's own label token over the whole vocabulary (`identity.log_odds`, the share's unit)
  lp        log P of that token (the old steering columns)
  contrast  logit A - logit B signed toward the window's own label (E112's margin): the contrast between the two labels

The share (do-nothing route split, `local_faith.route_split` with the components left as released): r = mean(L01 - L00) /
mean(T1 - T0), L the CI-masked model (Delta out) with Gamma's gates from W0 (L00) or from the state (L01), T the target
model at W0 (T0) or the state (T1). Same calls as `route_cells` and `masked_lo` (bf16 autocast, `compute_ci(chunk=B)`,
batches of 64), so on the same windows and batches its lo cells are theirs.

The neuron version (item 7): the original model with each block's firing (down_proj's input, `steering.Net` site mlp<l>)
masked to a TopK set, at every position: N01 with the k neurons of largest |firing| in the state's own forward, N00 with
W0's own. Its share is mean(N01 - N00) / mean(T1 - T0), T read by `steering.Net` (fp32).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from commute import steering as ST
from commute import vpd
from commute.faithfulness import model_state
from commute.identity import log_odds
from commute.local_faith import masked_logits

UNITS = ("lo", "lp", "contrast")
BATCH = 64
Cells = dict[str, dict[str, Tensor]]  # cell -> unit -> (N,) float64 CPU


def units(lg: Tensor, y: Tensor, ab: tuple[int, int]) -> dict[str, Tensor]:
    """Logits at the answer (N, V) -> (N,) float64 CPU per unit."""
    lg, y = lg.float(), y.to(lg.device)
    tgt = torch.where(y == 1, ab[0], ab[1])
    lp = torch.log_softmax(lg, -1).gather(1, tgt[:, None]).squeeze(1)
    return {"lo": log_odds(lg, tgt).double().cpu(), "lp": lp.double().cpu(),
            "contrast": ((lg[:, ab[0]] - lg[:, ab[1]]) * (2 * y - 1)).double().cpu()}  # fmt: skip


def cat(parts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {u: torch.cat([p[u] for p in parts]) for u in parts[0]}


def share(num: Tensor, gain: Tensor, floor: float) -> float:
    """mean(num) / mean(gain); nan when the mean gain is below `floor` (the share is not read below it)."""
    g = float(gain.mean())
    return float(num.mean()) / g if g >= floor and g != 0 else float("nan")


UnitsOf = Callable[[Tensor, slice], dict[str, Tensor]]  # (logits at the answer (B, V), the batch's rows) -> unit -> (B,)


def label_units(y: Tensor, ab: tuple[int, int]) -> UnitsOf:
    """`units` of the rows' own labels y, for the `_by` reads."""
    return lambda lg, rows: units(lg, y[rows], ab)


@torch.no_grad()
def share_cells(model: object, W0: Mapping[str, Tensor], dW: ST.DW, toks: Tensor, y: Tensor, ab: tuple[int, int], pos: int) -> Cells:
    """T0, T1, L00, L01 per window and unit on `toks` (N, L), the label read at `pos` (see the module docstring)."""
    return share_cells_by(model, W0, dW, toks, pos, label_units(y, ab))


@torch.no_grad()
def share_cells_by(model: object, W0: Mapping[str, Tensor], dW: ST.DW, toks: Tensor, pos: int, units_of: UnitsOf) -> Cells:
    """`share_cells` with any units (`units_of`), e.g. log-odds of each window's next token (P2.38 run 2); same calls."""
    return route_cells_by(model, W0, dW, toks, pos, units_of)[0]


KLCells = dict[str, Tensor]  # E57's cell -> (N,) float64 CPU, each row's KL summed over its positions 0..upto


@torch.no_grad()
def route_cells_by(model: object, W0: Mapping[str, Tensor], dW: ST.DW, toks: Tensor, pos: int, units_of: UnitsOf,
                   kl_upto: int | None = None, comps1: dict[str, tuple[Tensor, Tensor]] | None = None) -> tuple[Cells, KLCells]:
    """`share_cells_by`'s cells (T0, T1, L00, L01 at `pos`) and, with kl_upto, from the same forwards E57's KL cells
    (experiments/qwen_reading/lab.py `cells`, the form of Qwen's re-gating share, `qwen_reading.shares`) on the 67M, each
    row's KL summed over positions 0..kl_upto: K0 = KL(T0 || L00), K00 = KL(T1 || L00), K01 = KL(T1 || L01), T1 the
    changed model's own forward, L the CI-masked model (Delta out) with the released components and Gamma's gates from W0
    (L00) or from the changed model (L01); with comps1 (the change's split applied to the components) also K10 and K11,
    the masked model on comps1 under W0's gates (K10) and under the gates of the changed model carrying comps1 (K11),
    the 67M route split's convention (`local_faith.route_split`: L10 on W0's gates, L11 the state's own). Batches of
    BATCH, bf16-autocast masked forwards, fp32 KL (`vpd.kl_per_position`)."""
    out: dict[str, list[dict[str, Tensor]]] = {c: [] for c in ("T0", "L00", "T1", "L01")}
    names = ("K0", "K00", "K01") + (("K10", "K11") if comps1 is not None else ())
    kl: dict[str, list[Tensor]] = {k: [] for k in names} if kl_upto is not None else {}
    device = next(model.parameters()).device  # type: ignore[attr-defined]
    W1 = {k: w + dW[k].to(w.device, w.dtype) for k, w in W0.items()}
    for s in range(0, len(toks), BATCH):
        x, rows = toks[s : s + BATCH].to(device), slice(s, s + BATCH)
        lg: dict[str, Tensor] = {}
        g: dict[str, dict[str, Tensor]] = {}
        for T, L, W in (("T0", "L00", None), ("T1", "L01", W1)):
            with model_state(model, target=W):  # W None: the model as it stands, W0
                ci = vpd.compute_ci(model, x, chunk=x.shape[0])
                m = masked_logits(model, x, ci.g, None, None)
                out[T].append(units_of(ci.logits[:, pos], rows))
                out[L].append(units_of(m[:, pos], rows))
                if kl:
                    lg[T], lg[L], g[T] = ci.logits, m, ci.g
        if not kl:
            continue
        assert kl_upto is not None
        for name, (p, q) in {"K0": ("T0", "L00"), "K00": ("T1", "L00"), "K01": ("T1", "L01")}.items():
            kl[name].append(kl_rows(lg[p], lg[q], kl_upto))
        if comps1 is not None:
            with model_state(model, components=comps1):
                kl["K10"].append(kl_rows(lg["T1"], masked_logits(model, x, g["T0"], None, None), kl_upto))
                with model_state(model, target=W1):
                    g11 = vpd.compute_ci(model, x, chunk=x.shape[0]).g
                kl["K11"].append(kl_rows(lg["T1"], masked_logits(model, x, g11, None, None), kl_upto))
        del lg, g
    return {c: cat(v) for c, v in out.items()}, {k: torch.cat(v) for k, v in kl.items()}


def kl_rows(p: Tensor, q: Tensor, upto: int) -> Tensor:
    """Each row's KL(p || q) summed over positions 0..upto, (B,) float64 CPU (fp32 per position, `vpd.kl_per_position`)."""
    return vpd.kl_per_position(p[:, : upto + 1], q[:, : upto + 1]).double().sum(1).cpu()


def top_masks(firing: Mapping[str, Tensor], k: Mapping[int, int]) -> dict[int, Tensor]:
    """Per block l, (B, L, width) bool: True on the k[l] neurons of largest |firing| at each window and position."""
    out = {}
    for l, kl in k.items():
        a = firing[f"mlp{l}"]
        idx = a.abs().topk(min(kl, a.shape[-1]), dim=-1).indices
        out[l] = torch.zeros_like(a, dtype=torch.bool).scatter(-1, idx, True)
    return out


def top_overlap(f0: Mapping[str, Tensor], f1: Mapping[str, Tensor], k: Mapping[int, int], pos: int) -> dict[int, dict[str, Tensor]]:
    """Per block l, at position `pos` of each window, the `top_masks` sets of two forwards' firing (f0, f1 as `Net.run`
    captures them): "jaccard" |both| / |either| of the two sets, and the squared firing change (f1 - f0)^2 summed over the
    neurons in both sets ("d2_both") and over every neuron ("d2_all"); each (B,) float64 CPU. Identical firing reads
    jaccard exactly 1 and d2 exactly 0 (E155 neuron_overlap, cell P2.18)."""
    a = {f"mlp{l}": f0[f"mlp{l}"][:, pos] for l in k}
    b = {f"mlp{l}": f1[f"mlp{l}"][:, pos] for l in k}
    m0, m1 = top_masks(a, k), top_masks(b, k)
    out = {}
    for l in k:
        both, either = m0[l] & m1[l], m0[l] | m1[l]
        d2 = (b[f"mlp{l}"].double() - a[f"mlp{l}"].double()) ** 2
        out[l] = {"jaccard": (both.sum(-1).double() / either.sum(-1).double()).cpu(), "d2_both": (d2 * both).sum(-1).cpu(), "d2_all": d2.sum(-1).cpu()}
    return out


def mask_edit(masks: Mapping[int, Tensor]) -> ST.Edit:
    """Keep only the masked neurons' firing at every block in `masks`, every position."""
    return {f"mlp{l}": (lambda x, m=m: x * m.to(x.dtype)) for l, m in masks.items()}


@torch.no_grad()
def topk_logits(net: ST.Net, dW: ST.DW, b: Tensor, pos: int, k: Mapping[int, int]) -> dict[str, Tensor]:
    """One batch b (B, L): the logits at `pos` of T0, T1 (fp32 target model) and N00, N01 (W0 with firing masked to W0's
    or the state's TopK sets). Any task's readout is taken from these (E140's units, E143's log-odds of a target)."""
    lg0, f0 = net.run(b, capture=True)
    with net.at(dW):
        lg1, f1 = net.run(b, capture=True)
    out = {"T0": lg0[:, pos], "T1": lg1[:, pos]}
    for cell, f in (("N00", f0), ("N01", f1)):
        out[cell] = net.run(b, mask_edit(top_masks(f, k)))[0][:, pos]
    return out


def topk_cells(net: ST.Net, dW: ST.DW, toks: Tensor, y: Tensor, ab: tuple[int, int], pos: int, k: Mapping[int, int]) -> Cells:
    """T0, T1, N00, N01 (`topk_logits`) per window and unit."""
    return topk_cells_by(net, dW, toks, pos, k, label_units(y, ab))


def topk_cells_by(net: ST.Net, dW: ST.DW, toks: Tensor, pos: int, k: Mapping[int, int], units_of: UnitsOf) -> Cells:
    """`topk_cells` with any units (`units_of`, as `share_cells_by`)."""
    out: dict[str, list[dict[str, Tensor]]] = {c: [] for c in ("T0", "T1", "N00", "N01")}
    for s in range(0, len(toks), BATCH):
        for cell, lg in topk_logits(net, dW, toks[s : s + BATCH], pos, k).items():
            out[cell].append(units_of(lg, slice(s, s + BATCH)))
    return {c: cat(v) for c, v in out.items()}


@torch.no_grad()
def on_count(model: object, toks: Tensor, keys: tuple[str, ...]) -> tuple[dict[str, float], int]:
    """Gamma's gates above 0.5 at W0, summed over windows and positions per matrix in `keys`, and the positions counted
    (E136's `control_d`; k = `untrained.k_from_counts`)."""
    device = next(model.parameters()).device  # type: ignore[attr-defined]
    counts, n = dict.fromkeys(keys, 0.0), 0
    for s in range(0, len(toks), BATCH):
        x = toks[s : s + BATCH].to(device)
        g = vpd.compute_ci(model, x, chunk=x.shape[0]).g
        for key in keys:
            counts[key] += float((g[key] > 0.5).sum())
        n += x.shape[0] * x.shape[1]
    return counts, n
