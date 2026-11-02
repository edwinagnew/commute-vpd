"""E63's neuron-basis sparsity readings and its filed rules (`experiments/neuron_sparsity`; plan
reports/plans/2026-11-02_plan_e63-neuron-sparsity.md). No model is read here: tensors in, numbers out.

The matrices, read from the target checkpoint on 2026-11-02 (`h.<l>.mlp.*`, no bias terms):
  `h.<l>.mlp.c_fc`      the up-projection, weight (3072, 768); row i = neuron i's input weights
  `h.<l>.mlp.down_proj` the down-projection, weight (768, 3072); column i = neuron i's output weights
`steering.Net` calls neuron i of block l site `mlp<l>`, index i: the GELU output, which is down_proj's input. Flat neuron
index = l x 3072 + i, as `neurons.py`. A neuron's update = its row of delta-c_fc plus its column of delta-down_proj.

Rotating the neuron index by an orthogonal Q sends c_fc to Q c_fc, down_proj to down_proj Q^T and an activation a to Q a.
It is a change of reading, not a symmetry of the model: the GELU sits between c_fc and the neuron, so GELU(Q c_fc x) is
not Q GELU(c_fc x). A rotated value is therefore no ceiling: rotation keeps how an update divides among blocks and reads
its effective rank, and the GELU alone makes the neuron basis special for any update (review 2026-11-02, findings 1, 4).
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


def block_readings(dW: DW) -> dict[str, list[float]]:
    """Per block: its share of the update's energy, and W1 within the block as a fraction of 3072 (review, finding 4)."""
    e = neuron_energy(dW).split(D_MLP)
    total = sum(float(x.sum()) for x in e)
    return {"block_shares": [float(x.sum()) / total if total > 0 else math.nan for x in e],
            "block_W1": [float(participation(x)) / D_MLP for x in e]}


def random_ranks(seed: int) -> Tensor:
    """(12288,) one seeded random order of the neurons, as ranks: C1's reference curve patches rank < k on every window."""
    return torch.randperm(N_NEURONS, generator=torch.Generator().manual_seed(seed))


def window_ranks(shift: Tensor) -> Tensor:
    """(N, 12288) each neuron's rank (0 = largest) by that window's own |shift|; ties to the lower flat index."""
    order = torch.sort(shift.abs(), dim=1, descending=True, stable=True).indices
    ranks = torch.empty_like(order)
    return ranks.scatter_(1, order, torch.arange(shift.shape[1], device=shift.device).expand_as(order))


def median(x: Tensor) -> float:
    """The median ignoring nan (a window with no shift); the two middle values averaged; nan if all are nan."""
    return float(torch.nanquantile(x.double(), 0.5))


def half_share_index(shares: dict[int, float]) -> float:
    """C1's index: log2 k where the share curve first reaches half the share at the grid's last k (the full width), read
    by linear interpolation in log2 k between the grid points either side, so it is continuous. A crossing at the first
    k reads that k; a tie at half reads that grid point. nan if the full-width share is not a positive number, since then
    'half of it' is no target. With a positive full-width share the last k always qualifies, so a value is always found."""
    grid = sorted(shares)
    full = shares[grid[-1]]
    if not math.isfinite(full) or full <= 0:
        return math.nan
    half = full / 2
    i = next(i for i, k in enumerate(grid) if shares[k] >= half)
    if i == 0:
        return math.log2(grid[0])
    x0, x1, s0, s1 = math.log2(grid[i - 1]), math.log2(grid[i]), shares[grid[i - 1]], shares[grid[i]]
    return x0 + (half - s0) / (s1 - s0) * (x1 - x0)  # s1 >= half > s0, so s1 > s0


def c1_reading(shares: dict[int, float], G: float, G_score: float, floor: float) -> dict[str, Any]:
    """C1 with its readability: |G| and the full-width rise in nats must each reach `floor`; otherwise nan."""
    full = shares[max(shares)]
    readable = abs(G) >= floor and math.isfinite(full) and full > 0 and abs(full * G_score) >= floor
    return {"C1": half_share_index(shares) if readable else math.nan, "readable": bool(readable), "full_share": full}


def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x)


def separation(elicit: Sequence[float], new: Sequence[float], lower_is_sparser: bool, noise_floor: float = 0.0) -> dict[str, Any]:
    """E51's rule on the readable (finite) values: the gap of the kinds' means strictly over the larger within-kind range
    (or `noise_floor`, if larger), with the elicit side the sparser. Fewer than two readable on a side: 'unreadable', which
    does not pass."""
    a, b = [x for x in elicit if math.isfinite(x)], [x for x in new if math.isfinite(x)]
    rec: dict[str, Any] = {"n_elicit": len(a), "n_new": len(b)}
    if len(a) < 2 or len(b) < 2:
        return {**rec, "status": "unreadable", "pass": False}
    ma, mb = _mean(a), _mean(b)
    gap, spread = abs(ma - mb), max(max(a) - min(a), max(b) - min(b), noise_floor)
    sparser = ma < mb if lower_is_sparser else ma > mb
    ok = gap > spread and sparser
    return {**rec, "elicit_mean": ma, "new_mean": mb, "gap": gap, "spread": spread, "separates": gap > spread,
            "elicit_sparser": sparser, "status": "pass" if ok else "fail", "pass": ok}


def sparser_gap(elicit: Sequence[float], new: Sequence[float], lower_is_sparser: bool) -> float:
    """mean(new) - mean(elicit) with the sign that makes 'elicit sparser' positive; nan under two readable on a side."""
    a, b = [x for x in elicit if math.isfinite(x)], [x for x in new if math.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    return _mean(b) - _mean(a) if lower_is_sparser else _mean(a) - _mean(b)


def basis_guard(values: Sequence[float], reference: Sequence[float], lower_is_sparser: bool) -> dict[str, Any]:
    """The number's readable values against their reference, one-sided: the mean lies on the sparser side of the
    reference's mean, strictly by more than the values' range. A zero range passes if the mean is on that side at all.
    Under two readable values, or no finite reference: 'unreadable', which does not pass."""
    a, b = [x for x in values if math.isfinite(x)], [x for x in reference if math.isfinite(x)]
    if len(a) < 2 or not b:
        return {"status": "unreadable", "n": len(a), "pass": False}
    ma, mb = _mean(a), _mean(b)
    side = ma < mb if lower_is_sparser else ma > mb
    ok = side and abs(ma - mb) > max(a) - min(a)
    return {"n": len(a), "mean": ma, "reference_mean": mb, "gap": abs(ma - mb), "range": max(a) - min(a),
            "on_sparser_side": side, "status": "pass" if ok else "fail", "pass": ok}


def read_test(rows: dict[str, dict[str, Any]], num: str, spec: dict[str, Any], test: dict[str, list[str]]) -> dict[str, Any]:
    """One test on one number: `separation`, and its sparser gap strictly over the same signed gap on every reference,
    read on the states where the number is readable (a gap that survives the reference is not neuron information; a
    reference gap on the other side is beaten). A nan reference gap makes the test 'unreadable'. `reported` gaps gate nothing."""
    lower = spec["lower_is_sparser"]
    sides = {s: [n for n in test[s] if math.isfinite(rows[n][num])] for s in ("elicit", "new")}

    def gap(key: str) -> float:
        return sparser_gap([rows[n][key] for n in sides["elicit"]], [rows[n][key] for n in sides["new"]], lower)

    rec = separation([rows[n][num] for n in test["elicit"]], [rows[n][num] for n in test["new"]], lower, spec["noise_floor"])
    rec["sparser_gap"] = gap(num)
    rec["reference_gaps"] = {r: gap(r) for r in spec["references"]}
    rec["reported_gaps"] = {r: gap(r) for r in spec["reported"]}
    rec["beats_references"] = all(rec["sparser_gap"] > g for g in rec["reference_gaps"].values())
    if rec["status"] == "pass":
        if any(math.isnan(g) for g in rec["reference_gaps"].values()):
            rec["status"], rec["pass"] = "unreadable", False
        elif not rec["beats_references"]:
            rec["status"], rec["pass"] = "fail", False
    return rec


def score_number(rows: dict[str, dict[str, Any]], num: str, spec: dict[str, Any], tests: dict[str, Any], beside: dict[str, list[str]]) -> dict[str, Any]:
    """T1, T2, T3_B, T3_C and the guard against each reference on T1's elicit states; yes only if all pass. `beside` holds
    unscored values reported next to a test."""
    rec: dict[str, Any] = {k: read_test(rows, num, spec, t) for k, t in tests.items()}
    elicit = tests["T1"]["elicit"]
    values = [rows[n][num] for n in elicit]
    rec["guard"] = {r: basis_guard(values, [rows[n][r] for n in elicit], spec["lower_is_sparser"]) for r in spec["references"]}
    rec["beside"] = {k: {n: rows[n][num] for n in names} for k, names in beside.items()}
    rec["answers_yes"] = all(rec[k]["pass"] for k in tests) and all(g["pass"] for g in rec["guard"].values())
    return rec
