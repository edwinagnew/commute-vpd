"""Shared steps for reading E57's re-gating share on Qwen3-1.7B through the Lab in experiments/qwen_reading/lab.py,
and for placing a reading against a random-change curve at its own output size (cell P3.10). The reading steps were
first written in experiments/qwen_library_share/run.py (P3.8, P3.9), which stays as it ran.

The cells are E57's, per row: K0 (W0 against its own masked forward), K00 and K01 (the changed model against the
original subcomponents under W0's masks g0, and under the masks g1 of its own forward), K10 and K11 (the same with the
change's least-squares split applied to the live subcomponents). Share = (K00 - K01) / (K00 - K11) over the rows, with
`qwen_reading.shares`' status. A change's size is its K00 in mean nats per position over its rows.
Tests: tests/test_share_reading.py (the search and the placement rule; no model).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from transformers import AutoTokenizer

from commute import library_states as LS
from commute import qwen_pairs as P
from commute import qwen_reading as qr
from commute.qwen_d3 import null_seed, pinned
from commute.qwen_eval import MODEL, MODEL_REV
from commute.qwen_vpd import SITES

ROWSETS = ("copy", "cipher", "known", "invented", "gold")  # the pair library's row sets (gold's for both gold arms)


def rows_of(rowset: str) -> Tensor:
    """A row set as the library built it (128 x 256): a built side's packed trained cue sequences, or gold's rows (E40's
    held-out problems with gold solutions)."""
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    if rowset == "gold":
        return P.gold_rows(tok).tokens
    end_id = int(tok.convert_tokens_to_ids("<|im_end|>"))
    return P.own_rows(tok, end_id, rowset, P.tasks(LS.LIBRARY / "words.json")[rowset]["train"]).tokens


# ----------------------------------------------------------------------------- the reading (GPU, through the Lab)


def total(rows: dict[str, list[float]]) -> dict[str, float]:
    return {k: sum(v) for k, v in rows.items()}


def w0_control(lab: Any, name: str, r16: Tensor) -> dict[str, Any]:
    """W0 as the changed model: K00 = K01 = K0, and under an all-zero split K10 = K11 = K0 (within id_tol, relative)."""
    lab.set_weights("FT", None)
    w0 = total(lab.cells(r16, tag=f"W0 as FT, {name}"))
    zero = total(lab.cells(r16, lab.change(None, "zero"), tag=f"W0, zero split, {name}"))
    lab.drop_set("FT")
    tol = lab.cfg.id_tol
    holds = all(qr.close(w0[k], w0["K0"], tol) for k in ("K00", "K01")) and all(
        qr.close(zero[k], w0["K0"], tol) for k in ("K10", "K11"))
    return lab.check(f"W0_{name}", holds, {"w0": w0, "zero_split": zero})


def do_nothing(lab: Any, dW: dict[str, Tensor], r16: Tensor) -> dict[str, Any]:
    """The do-nothing state of dW through the split path: K10 = K00 and K11 = K01 by construction, so the share is 1."""
    lab.set_weights("FT", dW)
    cells = total(lab.cells(r16, tag="do-nothing")) | total(lab.cells(r16, lab.change(dW, None), tag="do-nothing split"))
    lab.drop_set("FT")
    share = qr.shares(cells, r16.numel())["regating"]
    return lab.check("do_nothing", qr.ok(share) and abs(share - 1) <= lab.cfg.id_tol, {"share": share} | cells)


def read(lab: Any, name: str, dW: dict[str, Tensor], X: dict[str, Tensor], rows: Tensor, floor: float,
         extra: dict[str, Any]) -> dict[str, Any]:
    """The four cells of W0 + dW on `rows`, its split X applied for K10 and K11; the mean K0 must equal the row set's
    floor within rep_tol (relative), or the job stops. Written to <results>/<name>.json with `extra`."""
    lab.set_weights("FT", dW)
    cells = lab.cells(rows, tag=name)
    k0 = sum(cells["K0"]) / rows.numel()
    lab.check(f"K0_{name}", qr.close(k0, floor, lab.cfg.rep_tol), {"K0": k0, "floor": floor})
    cells |= lab.cells(rows, lab.change(dW, X), tag=f"{name}, split")
    lab.drop_set("FT")
    rec = {"rows": cells, "means": {k: sum(v) / rows.numel() for k, v in cells.items()}}
    rec |= qr.whole(cells, lab.cfg.row_len) | extra
    lab.put(name, rec)
    lab.log(f"[share] {name}: {rec['regating']:.9g} ({rec['status']}), K00 {rec['means']['K00']:.6g}, {lab.peaks()}")
    return rec


def excess(lab: Any, dW: dict[str, Tensor], rows: Tensor, scale: float = 1.0) -> float:
    """K00 - K0 of W0 + scale·dW, mean nats per position over `rows`: the change's output size above W0's own masked
    error (one pass of the cells, no split). The size search's measure."""
    lab.set_weights("FT", dW, scale)
    c = total(lab.cells(rows, tag=f"size, scale {scale:.6g}"))
    lab.drop_set("FT")
    return (c["K00"] - c["K0"]) / rows.numel()


def with_zeros(part: dict[str, Tensor], shapes: dict[str, tuple[int, ...]]) -> dict[str, Tensor]:
    """A change over every site of SITES: `part`'s matrices, and one shared zero tensor per shape elsewhere."""
    zeros = {sh: torch.zeros(sh) for sh in {shapes[s] for s in SITES if s not in part}}
    return {s: part[s] if s in part else zeros[shapes[s]] for s in SITES}


def gaussian(norms: dict[str, float], shapes: dict[str, tuple[int, ...]], draw: int) -> dict[str, Tensor]:
    """iid Gaussian matrices at the sites of `norms` (seed `null_seed(draw, site)`), each scaled to norms[site]
    (Frobenius, float64, then fp32)."""
    out = {}
    for s, n in norms.items():
        G = torch.randn(shapes[s], generator=torch.Generator().manual_seed(null_seed(draw, s)), dtype=torch.float64)
        out[s] = pinned((G * (n / float(G.norm()))).float())
    return out


# ----------------------------------------------------------------------------- the size search (no model)


def size_search(f: Callable[[float], float], target: float, x0: float, power: float, match: float, evals: int,
                cap: float = math.inf, whole: bool = False) -> dict[str, Any]:
    """The size parameter x (a scale, power 2; or a neuron count, power 1 and `whole`) with f(x) within the factor
    `match` of target, for f increasing in x and about ∝ x^power. Next x = x·(target/f)^(1/power), each step within
    ×/÷ 100, at most cap, and strictly inside the bracket of evaluated x below and above target (its geometric middle,
    or half the upper end, when the step leaves it). f ≤ 0 counts as below target, NaN or infinity as above. Stops when
    matched, after `evals` evaluations, or when no new x is left. The evaluated x whose f is closest in ratio is kept."""
    assert 0 < x0 <= cap and power > 0, (x0, cap, power)
    lo, hi, x, out = 0.0, math.inf, x0, []
    for _ in range(evals if qr.ok(target) and target > 0 else 0):
        y = f(x)
        out.append([x, y])
        if qr.within(y, target, match):
            break
        below = qr.ok(y) and y < target  # y ≤ 0 is below; NaN or infinity above
        lo, hi = (max(lo, x), hi) if below else (lo, min(hi, x))
        step = (target / y) ** (1 / power) if qr.ok(y) and y > 0 else (100.0 if below else 0.01)
        nxt = min(cap, x * min(100.0, max(0.01, step)))
        if not lo < nxt < hi:
            if hi == math.inf:
                break  # at the cap and still below target
            nxt = math.sqrt(lo * hi) if lo > 0 else hi / 2
        if whole:
            nxt = float(max(1, round(nxt)))
        if not lo < nxt < hi:
            break
        x = nxt
    gap = lambda e: abs(math.log(e[1] / target)) if qr.ok(e[1]) and e[1] > 0 else math.inf
    best = min(out, key=gap, default=[x0, qr.NAN])
    return {"target": target, "evals": out, "x": best[0], "f": best[1], "matched": qr.within(best[1], target, match)}


# ----------------------------------------------------------------------------- the placement rule (no model)


def curve(draws: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """The random curve of one row set: per size level with at least two usable draws, (the draws' geometric mean
    K00, their mean share), in increasing K00. A draw is {"level", "k00", "share", "usable"}; usable = scored, finite
    and matched in size."""
    by: dict[Any, list[dict[str, Any]]] = {}
    for d in draws:
        if d["usable"] and qr.ok(d["share"]) and qr.ok(d["k00"]) and d["k00"] > 0:
            by.setdefault(d["level"], []).append(d)
    pts = [(math.exp(sum(math.log(d["k00"]) for d in ds) / len(ds)), sum(d["share"] for d in ds) / len(ds))
           for ds in by.values() if len(ds) >= 2]
    return sorted(pts)


def at(cv: list[tuple[float, float]], k00: float | None, slack: float = 1.1) -> float | None:
    """The curve at k00, linear in log K00 between levels; a k00 outside the span by at most the factor `slack` takes
    the nearest end's share. None: fewer than two levels, k00 missing or not positive, or further outside."""
    if len(cv) < 2 or k00 is None or not qr.ok(k00) or k00 <= 0:
        return None
    (x0, y0), (xn, yn) = cv[0], cv[-1]
    if qr.r9(k00) < qr.r9(x0 / slack) or qr.r9(k00) > qr.r9(xn * slack):
        return None
    if k00 <= x0:
        return y0
    if k00 >= xn:
        return yn
    i = next(i for i in range(1, len(cv)) if k00 <= cv[i][0])
    (xa, ya), (xb, yb) = cv[i - 1], cv[i]
    t = math.log(k00 / xa) / math.log(xb / xa)
    return ya + t * (yb - ya)


def band(cv: list[tuple[float, float]], draws: list[dict[str, Any]]) -> float | None:
    """The largest distance of a usable random draw's share from the curve at its own K00; None if the curve has
    fewer than two levels."""
    devs = [abs(d["share"] - c) for d in draws if d["usable"] and qr.ok(d["share"])
            and (c := at(cv, d["k00"])) is not None]
    return max(devs) if len(cv) >= 2 and devs else None


def place(share: float | None, usable: bool, k00: float | None, cv: list[tuple[float, float]],
          bnd: float | None) -> dict[str, Any]:
    """One reading against its row set's curve: gap = share - curve at its K00; 'above' if gap > band, 'below' if
    gap < -band, 'inside' otherwise (rounded to 9 places, so a gap equal to the band is inside); 'unreadable' if the
    reading is not usable, its share not finite, the curve has no value at its K00 or there is no band."""
    c = at(cv, k00)
    if not usable or share is None or not qr.ok(share) or c is None or bnd is None:
        return {"curve": c, "gap": None, "band": bnd, "class": "unreadable"}
    gap = share - c
    cls = "above" if qr.r9(gap) > qr.r9(bnd) else "below" if qr.r9(gap) < -qr.r9(bnd) else "inside"
    return {"curve": c, "gap": gap, "band": bnd, "class": cls}


def outcome(classes: list[str], min_readable: int, frac: float = 0.8) -> str:
    """'above random', 'like random' or 'below random' when at least `frac` of the readable readings (at least
    min_readable of them) are 'above', 'inside' or 'below'; 'mixed' otherwise; 'unreadable' under min_readable."""
    readable = [c for c in classes if c != "unreadable"]
    if len(readable) < min_readable or not readable:
        return "unreadable"
    for cls, name in (("above", "above random"), ("inside", "like random"), ("below", "below random")):
        if qr.r9(readable.count(cls)) >= qr.r9(frac * len(readable)):
            return name
    return "mixed"
