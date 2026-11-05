"""E90 Amendment 2's filed rules (results/predictions_amend2.json; plan reports/plans/2026-11-05_plan_e90-amend2.md): random bases
matched in variance. Pure: no model is read, and tests/test_new_vs_reuse_amend2.py runs each rule through its edge cases. The band is
`removal.band_of` and `removal.max_band`, the verdict `rules.verdict`; only the draws change, plus one outcome, "no matched band possible"."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import rules
import torch
from torch import Tensor

from commute import removal as RM

NO_MATCH = "no matched band possible"
MATCH_RATIO = 0.9  # Fable's "at least 0.9 of the basis's"; the same ratio reads the task windows
ABOVE, WITHIN, MIXED, UNREADABLE = "above", "within", "mixed", "unreadable"
NO_READING, ALL_WITHIN, COPY_SPECIFIC, CONTROL_FAILS, OTHER = "no reading", "(i)", "(ii)", "(iii)", "other"
VALUE_FORM = ("rank1", "cipher", "elicit", "ident32", "copy24", "rand32")


# ---- the draws ----


def spectrum(h: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues (descending, zero or above) and eigenvectors of the centred covariance of rows h (n, d), float64."""
    x = h.double() - h.double().mean(0)
    vals, vecs = torch.linalg.eigh(x.T @ x / len(x))
    return vals.flip(0).clamp(min=0), vecs.flip(1)


def rank_of(lam: Tensor) -> int:
    """Components above 1e-10 of the largest: fp32 residuals' rounding variance sits near 1e-14 of it."""
    return int((lam > lam[0] * 1e-10).sum()) if lam[0] > 0 else 0


def expected_share(lam: Tensor, r: int, k: int) -> float:
    """The mean variance share of a uniformly random rank-r subspace of the top k components: r/k x their share. It never rises
    with k, since the mean of the top k eigenvalues never does."""
    return r / k * float(lam[:k].sum() / lam.sum())


def share_of(lam: Tensor, q: Tensor) -> float:
    """The variance share of the orthonormal basis q (k, r), written in principal-component coordinates."""
    return float((lam[: q.shape[0]] * q.pow(2).sum(1)).sum() / lam.sum())


def draws_in(k: int, r: int, n: int, key: Sequence[int], device: Any) -> list[Tensor]:
    return [RM.random_basis(k, r, [*key, k, i]).double().to(device) for i in range(n)]


def largest_k(lam: Tensor, r: int, target: float, n: int, key: Sequence[int]) -> int:
    """The largest k > r whose n realised draws average at least `target`, else r, where every draw is the top-r subspace, which
    by Ky Fan holds at least the basis's share. Largest, not Fable's "smallest": see the filing's `departure`."""
    for k in range(rank_of(lam), r, -1):
        if expected_share(lam, r, k) >= target and sum(share_of(lam, q) for q in draws_in(k, r, n, key, lam.device)) / n >= target:
            return k
    return r


def overlap(a: Tensor, b: Tensor) -> float:
    """||a^T b||_F^2 / rank a, for orthonormal a and b of the same rank: 1 for one subspace, r/k on average for random ones."""
    return float((a.double().T @ b.double()).pow(2).sum() / a.shape[1])


def mean_pairwise(draws: Sequence[Tensor]) -> float:
    pairs = [overlap(a, b) for i, a in enumerate(draws) for b in draws[i + 1 :]]
    return sum(pairs) / len(pairs) if pairs else math.nan


def split_half(h: Tensor, y: Tensor) -> float:
    """The bar: the overlap of C's value-form basis fitted on each value's even windows and on its odd ones, the subspace's own
    sampling noise. Draws at least this alike are one subspace re-estimated. nan with under two values, or under two windows of one."""
    seen = y.unique()
    if len(seen) < 2 or int(torch.bincount(y).index_select(0, seen).min()) < 2:
        return math.nan
    half = lambda p: RM.centred_basis(torch.stack([h[y == v][p::2].double().mean(0) for v in seen], 1))
    return overlap(half(0), half(1))


def matched(h: Tensor, y: Tensor, u: Tensor, n: int, key: Sequence[int]) -> tuple[dict[str, Any], list[Tensor]]:
    """C's draws at one block, from X_C's fit residuals h (M, d), value index y and basis u (d, r): the record and the draws (d, r),
    fp32. `reason` not None is "no matched band possible": a residual with no variance or with rank under r gives no draws; k = r or
    a mean pairwise overlap not under the split-half bar (a nan bar included) gives draws that are still read and printed beside."""
    lam, vecs = spectrum(h)
    r, total = u.shape[1], float(lam.sum())
    rec: dict[str, Any] = {"r": r, "residual_rank": rank_of(lam), "k": None, "reason": None}
    if not total > 0:
        return {**rec, "reason": "the fit residual has no variance"}, []
    if r > rec["residual_rank"]:
        return {**rec, "reason": f"rank {r} exceeds the residual's rank {rec['residual_rank']}"}, []
    x = h.double() - h.double().mean(0)
    share = float((x @ u.double().to(x.device)).pow(2).sum() / x.pow(2).sum())
    k = largest_k(lam, r, MATCH_RATIO * share, n, key)
    qs = draws_in(k, r, n, key, lam.device)
    draws = [(vecs[:, :k] @ q).float() for q in qs]
    alike, bar = mean_pairwise(qs), split_half(h, y)
    reason = f"k = r = {r}: every draw spans one subspace" if k == r else None if alike < bar else f"mean pairwise overlap {alike:.4f}, bar {bar:.4f}"
    return {**rec, "k": k, "reason": reason, "basis_share": share, "target": MATCH_RATIO * share, "draw_shares": [share_of(lam, q) for q in qs], "pairwise_overlap": alike,
            "bar": bar, "overlap_with_basis": sum(overlap(u.to(d.device), d) for d in draws) / n, "chance_overlap": r / k, "top_r_share": float(lam[:r].sum()) / total}, draws  # fmt: skip


def task_match(basis: float, draws: float) -> str:
    """The draws' mean variance share on the task windows against the basis's: "under" or "over" outside the same 0.9 ratio both
    ways. Printed beside the verdict and never changes it."""
    if not basis > 0 or math.isnan(draws):
        return "no share"
    return "under" if draws < MATCH_RATIO * basis else "over" if MATCH_RATIO * draws > basis else "matched"


# ---- the verdict ----


def status(shares: Sequence[float], band: float, readable: bool) -> str:
    """A map's seeds against the band, strict as `rules.verdict`. A nan (no gain, the do-nothing state), no seeds, or a C without
    teeth or a matched band: unreadable, never "within"."""
    if not readable or not shares or math.isnan(band) or any(math.isnan(s) for s in shares):
        return UNREADABLE
    above = [s > band for s in shares]
    return ABOVE if all(above) else WITHIN if not any(above) else MIXED


def read_block(states: Mapping[str, Any], names: Sequence[str], b: int, teeth: bool, reason: str | None) -> dict[str, Any]:
    """One C at one block: the states `names` and their writes against the band over all of them. `verdicts` is the filed column,
    `beside` the same read with teeth and a matched band assumed; stage 2's saved `states` go through it as well."""
    per: dict[str, list[str]] = {}
    beside: dict[str, list[str]] = {}
    shares: dict[str, list[float]] = {}
    at = {n: (states[n].get(str(b), {}), states[f"{n}_write"].get(str(b), {})) for n in names}
    band = RM.max_band([x.get("band", math.nan) for pair in at.values() for x in pair])
    for n, (s, w) in at.items():
        m, a, c = n.rsplit("_", 1)[0], s.get("A_lo", math.nan), w.get("A_lo", math.nan)
        per.setdefault(m, []).append(NO_MATCH if reason else rules.verdict(teeth, a, c, band))
        beside.setdefault(m, []).append(rules.verdict(True, a, c, band))
        shares.setdefault(m, []).append(a)
    return {"block": b, "band": band, "teeth": teeth, "reason": reason, "shares": shares, "verdicts": {m: rules.over_seeds(v) for m, v in per.items()},
            "beside": {m: rules.over_seeds(v) for m, v in beside.items()}, "status": {m: status(v, band, teeth and not reason) for m, v in shares.items()}}  # fmt: skip


def branch(st: Mapping[str, str]) -> dict[str, Any]:
    """Fable's branches on the maps' statuses at one block, first match wins. None readable: no reading. (i) every value-form map
    within. (iii) rand32 above: token identity's removal fails its negative control, and the identity-C rows carry that beside their
    verdict. (ii) rand32 within, ident32 and copy-24 above. Anything else: no branch, the per-row verdicts stand."""
    got = [st[m] for m in VALUE_FORM]
    if all(s == UNREADABLE for s in got):
        return {"branch": NO_READING, "beside": {}}
    if all(s == WITHIN for s in got):
        return {"branch": ALL_WITHIN, "beside": {}}
    if st["rand32"] == ABOVE:
        return {"branch": CONTROL_FAILS, "beside": {m: "removal not specific to C: rand32 above the band" for m in ("ident32", "copy24")}}
    if st["rand32"] == WITHIN and st["ident32"] == ABOVE and st["copy24"] == ABOVE:
        return {"branch": COPY_SPECIFIC, "beside": {}}
    return {"branch": OTHER, "beside": {}}


def named(per_block: Mapping[int, str]) -> str:
    """Each verdict names its block: "reuses C at block 1; does not reach at block 2"."""
    return "; ".join(f"{v} at block {b}" for b, v in sorted(per_block.items()))
