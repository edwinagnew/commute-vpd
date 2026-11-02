"""E63's neuron-basis sparsity readings and its filed rules (`experiments/neuron_sparsity`; plan
reports/plans/2026-11-02_plan_e63-neuron-sparsity.md). No model is read here: tensors in, numbers out.

The matrices, read from the target checkpoint on 2026-11-02 (`h.<l>.mlp.*`, no bias terms):
  `h.<l>.mlp.c_fc`      the up-projection, weight (3072, 768); row i = neuron i's input weights
  `h.<l>.mlp.down_proj` the down-projection, weight (768, 3072); column i = neuron i's output weights
`steering.Net` calls neuron i of block l site `mlp<l>`, index i: the GELU output, which is down_proj's input. Flat neuron
index = l x 3072 + i, as `neurons.py`. A neuron's update = its row of delta-c_fc plus its column of delta-down_proj.

Rotating the neuron index by an orthogonal Q (a' = Q a) sends c_fc to Q c_fc and down_proj to down_proj Q^T, and leaves
the model's function unchanged, so a reading that moves under Q reads something about the neuron basis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from commute.steering import D_MLP, DW, N_LAYERS

N_NEURONS = N_LAYERS * D_MLP


def up_key(layer: int) -> str:
    return f"h.{layer}.mlp.c_fc"


def down_key(layer: int) -> str:
    return f"h.{layer}.mlp.down_proj"


def mlp_part(dW: DW) -> DW:
    """The eight matrices the neuron readings use, in a fixed order (the Gaussian's draw depends on it)."""
    return {k: dW[k] for layer in range(N_LAYERS) for k in (up_key(layer), down_key(layer))}


def participation(e: Tensor) -> Tensor:
    """(sum e)^2 / sum e^2 over the last dimension; nan where every e is 0 (0 / 0: W0's reading, undefined)."""
    e = e.double()
    return e.sum(-1).pow(2) / e.pow(2).sum(-1)


def rotations(seed: int) -> list[Tensor]:
    """One fixed random orthogonal 3072 x 3072 matrix per block (QR of a seeded Gaussian, float64, on CPU)."""
    out = []
    for layer in range(N_LAYERS):
        g = torch.Generator().manual_seed(seed + layer)
        out.append(torch.linalg.qr(torch.randn(D_MLP, D_MLP, generator=g, dtype=torch.float64)).Q)
    return out


def rotate_shift(shift: Tensor, rot: list[Tensor]) -> Tensor:
    """(N, 12288) activation shifts in the rotated neuron index, block by block: a' = Q a."""
    return torch.cat([shift[:, i * D_MLP : (i + 1) * D_MLP] @ rot[i].T for i in range(N_LAYERS)], 1)


def neuron_energy(dW: DW, rot: list[Tensor] | None = None) -> Tensor:
    """(12288,) float64: each neuron's squared update, its row of delta-c_fc plus its column of delta-down_proj."""
    out = []
    for i in range(N_LAYERS):
        up, down = dW[up_key(i)].double(), dW[down_key(i)].double()
        if rot is not None:
            up, down = rot[i] @ up, down @ rot[i].T
        out.append(up.pow(2).sum(1) + down.pow(2).sum(0))
    return torch.cat(out)


def w1(dW: DW, rot: list[Tensor] | None = None) -> float:
    """W1 (and W1g on a gradient): participation ratio over neurons of the squared update, as a fraction of 12288."""
    return float(participation(neuron_energy(dW, rot))) / N_NEURONS


def w2(dW: DW, W0: DW) -> dict[str, float]:
    """W2: share of the update's energy along each neuron's own W0 row (`rows`), W0 column (`cols`), and both (`all`).
    A neuron whose W0 row or column is exactly 0 has no such direction and adds 0. nan when the update is 0."""
    along, total = {"rows": 0.0, "cols": 0.0}, {"rows": 0.0, "cols": 0.0}
    for i in range(N_LAYERS):
        for part, key, dim in (("rows", up_key(i), 1), ("cols", down_key(i), 0)):
            d, w = dW[key].double(), W0[key].double()
            norm2 = w.pow(2).sum(dim)
            proj = (d * w).sum(dim).pow(2) / norm2
            along[part] += float(torch.where(norm2 > 0, proj, torch.zeros_like(proj)).sum())
            total[part] += float(d.pow(2).sum())

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else math.nan

    return {"rows": ratio(along["rows"], total["rows"]), "cols": ratio(along["cols"], total["cols"]),
            "all": ratio(along["rows"] + along["cols"], total["rows"] + total["cols"])}


def sixteen_update(W0: DW, layer: int, neurons: Tensor) -> DW:
    """The positive control: W0's rows and columns of `neurons` in block `layer` rescaled by a_i, the other matrices 0.
    a_i makes each neuron's update energy equal (W1 = len / 12288 exactly); the mean of a_i is 1 (a doubling on average)."""
    up, down = W0[up_key(layer)].double(), W0[down_key(layer)].double()
    idx = neurons.to(up.device)
    a = 1 / (up[idx].pow(2).sum(1) + down[:, idx].pow(2).sum(0)).sqrt()
    a = a / a.mean()
    dW = {k: torch.zeros_like(w) for k, w in W0.items()}
    dW[up_key(layer)][idx] = (a[:, None] * up[idx]).to(W0[up_key(layer)].dtype)
    dW[down_key(layer)][:, idx] = (down[:, idx] * a[None, :]).to(W0[down_key(layer)].dtype)
    return dW


def window_ranks(shift: Tensor) -> Tensor:
    """(N, 12288) each neuron's rank (0 = largest) by that window's own |shift|; ties to the lower flat index."""
    order = torch.sort(shift.abs(), dim=1, descending=True, stable=True).indices
    ranks = torch.empty_like(order)
    return ranks.scatter_(1, order, torch.arange(shift.shape[1], device=shift.device).expand_as(order))


def median(x: Tensor) -> float:
    """The median ignoring nan (a window with no shift); the two middle values averaged; nan if all are nan."""
    return float(torch.nanquantile(x.double(), 0.5))


def half_share_index(shares: dict[int, float]) -> float:
    """C1's index: log2 of the first k on the grid (ascending) whose share >= half the share at the grid's last k (the
    full width); a tie at half counts. nan if the full-width share is not a positive number, since then 'half of it' is
    no target. With a positive full-width share the last k always qualifies, so a value is always found."""
    grid = sorted(shares)
    full = shares[grid[-1]]
    if not math.isfinite(full) or full <= 0:
        return math.nan
    return math.log2(next(k for k in grid if shares[k] >= full / 2))


def c1_reading(shares: dict[int, float], G: float, G_score: float, floor: float) -> dict[str, Any]:
    """C1 with its readability: |G| and the full-width rise in nats must each reach `floor`; otherwise nan."""
    full = shares[max(shares)]
    readable = abs(G) >= floor and math.isfinite(full) and full > 0 and abs(full * G_score) >= floor
    return {"C1": half_share_index(shares) if readable else math.nan, "readable": bool(readable), "full_share": full}


def separation(elicit: Sequence[float], new: Sequence[float], lower_is_sparser: bool) -> dict[str, Any]:
    """E51's rule on the readable (finite) values: the gap of the kinds' means strictly over the larger within-kind
    range, with the elicit side the sparser. Fewer than two readable on a side: 'unreadable', which does not pass."""
    a, b = [x for x in elicit if math.isfinite(x)], [x for x in new if math.isfinite(x)]
    rec: dict[str, Any] = {"n_elicit": len(a), "n_new": len(b)}
    if len(a) < 2 or len(b) < 2:
        return {**rec, "status": "unreadable", "pass": False}
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    gap, spread = abs(ma - mb), max(max(a) - min(a), max(b) - min(b))
    sparser = ma < mb if lower_is_sparser else ma > mb
    ok = gap > spread and sparser
    return {**rec, "elicit_mean": ma, "new_mean": mb, "gap": gap, "spread": spread, "separates": gap > spread,
            "elicit_sparser": sparser, "status": "pass" if ok else "fail", "pass": ok}


def basis_guard(values: Sequence[float], reference: Sequence[float]) -> dict[str, Any]:
    """The number's readable values against their reference (rotated, Gaussian or full width): the means differ strictly
    by more than the values' range. A zero range passes if the means differ at all. Under two readable: 'unreadable'."""
    a, b = [x for x in values if math.isfinite(x)], [x for x in reference if math.isfinite(x)]
    if len(a) < 2 or not b:
        return {"status": "unreadable", "n": len(a), "pass": False}
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    ok = abs(ma - mb) > max(a) - min(a)
    return {"n": len(a), "mean": ma, "reference_mean": mb, "gap": abs(ma - mb), "range": max(a) - min(a),
            "below_reference": ma < mb, "status": "pass" if ok else "fail", "pass": ok}
