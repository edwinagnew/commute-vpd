"""What a fixed change of Gamma's values is built from (E30, gamma_shift_map): pure bookkeeping on a shift s, one (C,)
vector per matrix (E18's fixed change, `regating.shift_of`), with dead subcomponents at 0.

Counts of alive entries above a size, the split of sum |s| by layer and by sign, the shift kept on one layer or on its
k largest entries pooled over every matrix, a rank correlation, and the update's write energy on each subcomponent's
read/write pair (|u_c^T dW v_c| / (|u_c| |v_c|), vpd's layout: U (C, d_out), V (d_in, C), dW (d_out, d_in)).
"""

from __future__ import annotations

import torch
from torch import Tensor

from commute.recruit import layer_or_none

Shift = dict[str, Tensor]


def pooled(d: Shift, alive: dict[str, Tensor]) -> Tensor:
    """The alive entries of every matrix, concatenated in sorted key order (float64)."""
    return torch.cat([d[k].double().cpu()[alive[k].cpu().bool()] for k in sorted(d)])


def counts(s: Shift, alive: dict[str, Tensor], thresholds: tuple[float, ...]) -> dict[str, int]:
    """Alive total, entries with |s| > each threshold, and the numbers turned up (s > 0) and down (s < 0)."""
    v = pooled(s, alive)
    out = {"alive": int(v.numel()), "up": int((v > 0).sum()), "down": int((v < 0).sum())}
    out.update({f"abs_gt_{t:g}": int((v.abs() > t).sum()) for t in thresholds})
    return out


def mass_split(s: Shift) -> dict[str, dict[str, float]]:
    """Shares of sum |s| per layer and per sign (turned up, turned down); each set of shares sums to 1."""
    total = sum(float(v.abs().sum()) for v in s.values())
    layer: dict[str, float] = {}
    for k, v in s.items():
        name = str(layer_or_none(k))
        layer[name] = layer.get(name, 0.0) + float(v.abs().sum())
    sign = {"up": sum(float(v.clamp(min=0).sum()) for v in s.values()),
            "down": sum(float(-v.clamp(max=0).sum()) for v in s.values())}
    share = (lambda x: x / total) if total else (lambda _x: float("nan"))
    return {"by_layer": {k: share(v) for k, v in sorted(layer.items())}, "by_sign": {k: share(v) for k, v in sign.items()}}


def on_layer(s: Shift, layer: int) -> Shift:
    """s on the matrices of one layer, 0 elsewhere."""
    return {k: v if layer_or_none(k) == layer else torch.zeros_like(v) for k, v in s.items()}


def top_k(s: Shift, k: int) -> Shift:
    """s on its k entries of largest |s| over every matrix together (ties by flat order), 0 elsewhere."""
    keys = sorted(s)
    flat = torch.cat([s[key].flatten() for key in keys])
    keep = torch.zeros_like(flat, dtype=torch.bool)
    keep[flat.abs().argsort(descending=True, stable=True)[: min(k, flat.numel())]] = True
    out, i = {}, 0
    for key in keys:
        n = s[key].numel()
        out[key] = torch.where(keep[i : i + n].view_as(s[key]), s[key], torch.zeros_like(s[key]))
        i += n
    return out


def spearman(x: Tensor, y: Tensor) -> float:
    """Rank correlation (ranks by argsort, ties broken by order); nan if either side is constant."""
    rx, ry = (t.double().argsort().argsort().double() for t in (x, y))
    if float(x.double().std()) == 0 or float(y.double().std()) == 0:
        return float("nan")
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def write_energy(uv: dict[str, tuple[Tensor, Tensor]], dW: dict[str, Tensor]) -> Shift:
    """Per matrix (C,): |u_c^T dW v_c| / (|u_c| |v_c|), the update's size along subcomponent c's own rank-1 direction."""
    out = {}
    for k, (U, V) in uv.items():
        U, V, M = U.double().cpu(), V.double().cpu(), dW[k].double().cpu()
        out[k] = (((U @ M) * V.T).sum(1).abs() / (U.norm(dim=1) * V.norm(dim=0)).clamp_min(1e-30)).float()
    return out


def opened(g: dict[str, Tensor], shift: Shift) -> Tensor:
    """Per position, summed over every matrix's subcomponents: how far the shift's turned-down entries lower Gamma's
    values, g - clip(g + min(s, 0), 0, 1) = min(g, |s|) where s < 0. The adversary's range [Gamma', 1] widens by this
    much; a turned-down entry on a subcomponent already at 0 opens nothing (E30 amendment 1). g: (..., C) per matrix."""
    per = [torch.minimum(v.float(), (-shift[k].to(v.device).float()).clamp(min=0.0)).sum(-1) for k, v in g.items()]
    return torch.stack(per).sum(0)
