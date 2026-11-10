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

from collections.abc import Mapping

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
    return {u: torch.cat([p[u] for p in parts]) for u in UNITS}


def share(num: Tensor, gain: Tensor, floor: float) -> float:
    """mean(num) / mean(gain); nan when the mean gain is below `floor` (the share is not read below it)."""
    g = float(gain.mean())
    return float(num.mean()) / g if g >= floor and g != 0 else float("nan")


@torch.no_grad()
def share_cells(model: object, W0: Mapping[str, Tensor], dW: ST.DW, toks: Tensor, y: Tensor, ab: tuple[int, int], pos: int) -> Cells:
    """T0, T1, L00, L01 per window and unit on `toks` (N, L), the label read at `pos` (see the module docstring)."""
    out: dict[str, list[dict[str, Tensor]]] = {c: [] for c in ("T0", "L00", "T1", "L01")}
    device = next(model.parameters()).device  # type: ignore[attr-defined]
    W1 = {k: w + dW[k].to(w.device, w.dtype) for k, w in W0.items()}
    for s in range(0, len(toks), BATCH):
        x, yb = toks[s : s + BATCH].to(device), y[s : s + BATCH]
        for T, L, W in (("T0", "L00", None), ("T1", "L01", W1)):
            with model_state(model, target=W):  # W None: the model as it stands, W0
                ci = vpd.compute_ci(model, x, chunk=x.shape[0])
                out[T].append(units(ci.logits[:, pos], yb, ab))
                out[L].append(units(masked_logits(model, x, ci.g, None, None)[:, pos], yb, ab))
    return {c: cat(v) for c, v in out.items()}


def top_masks(firing: Mapping[str, Tensor], k: Mapping[int, int]) -> dict[int, Tensor]:
    """Per block l, (B, L, width) bool: True on the k[l] neurons of largest |firing| at each window and position."""
    out = {}
    for l, kl in k.items():
        a = firing[f"mlp{l}"]
        idx = a.abs().topk(min(kl, a.shape[-1]), dim=-1).indices
        out[l] = torch.zeros_like(a, dtype=torch.bool).scatter(-1, idx, True)
    return out


def mask_edit(masks: Mapping[int, Tensor]) -> ST.Edit:
    """Keep only the masked neurons' firing at every block in `masks`, every position."""
    return {f"mlp{l}": (lambda x, m=m: x * m.to(x.dtype)) for l, m in masks.items()}


@torch.no_grad()
def topk_cells(net: ST.Net, dW: ST.DW, toks: Tensor, y: Tensor, ab: tuple[int, int], pos: int, k: Mapping[int, int]) -> Cells:
    """T0, T1 (fp32 target model) and N00, N01 (W0 with firing masked to W0's or the state's TopK sets) per window and unit."""
    out: dict[str, list[dict[str, Tensor]]] = {c: [] for c in ("T0", "T1", "N00", "N01")}
    for s in range(0, len(toks), BATCH):
        b, yb = toks[s : s + BATCH], y[s : s + BATCH]
        lg0, f0 = net.run(b, capture=True)
        with net.at(dW):
            lg1, f1 = net.run(b, capture=True)
        out["T0"].append(units(lg0[:, pos], yb, ab))
        out["T1"].append(units(lg1[:, pos], yb, ab))
        for cell, f in (("N00", f0), ("N01", f1)):
            out[cell].append(units(net.run(b, mask_edit(top_masks(f, k)))[0][:, pos], yb, ab))
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
