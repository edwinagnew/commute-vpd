"""Our vectorised path for WeightWatch's detector (P2.73), the check of the staged core against its source, the
known-range test (appendix C1) and the parity harness that runs the staged `LatentStats` beside our path.

Our path does what the code does (`ww_core.LatentStats.register` and `set_threshold`, threshold percentile 0):
  per window and direction u   s_min, s_max = min and max over tokens of cos(a_t, u)          (`window_stats`)
  pass 1   ranges from the first n - int(0.1 n) windows, FPR on the last int(0.1 n)               (`calibrate`)
  pass 2   ranges from all windows but the last: the installed ranges
  flag     s_min < c_min - EPS or s_max > c_max + EPS for any direction                            (`flagged`)
A range is the min of the s_min and the max of the s_max (np.percentile at 0 and 100 picks those elements exactly).
The slicing is the code's, `[:-v]`, so v = 0 (under 10 windows) leaves pass 1 empty and its FPR 0, as there. Values
are compared in float64, as the code compares Python floats of the fp32 cosines.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

EPS = 0.01  # WW utils.py line 6; check_core asserts the staged core holds the same
HERE = Path(__file__).resolve().parent
CORE = HERE / "ww_core.py"
SOURCE_LINES = (156, 343)  # 1-based, inclusive: class LatentStats in WW src/utils.py
PATCH = ("self.datapoints.append(np.array(current_datapoint))", "self.datapoints.append(current_datapoint)")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_core(source: Path, source_sha: str, core_sha: str) -> list[str]:
    """Misses: the source's and the staged file's sha256 against the filed ones, and the staged class against source
    lines 156-343 with the one patch (applied exactly once). The source is read as text only."""
    miss = [f"{p.name}: sha256 {sha256(p)} != filed {w}" for p, w in ((source, source_sha), (CORE, core_sha)) if sha256(p) != w]
    lines = source.read_text().splitlines()
    want = "\n".join(lines[SOURCE_LINES[0] - 1 : SOURCE_LINES[1]])
    if want.count(PATCH[0]) != 1:
        return [*miss, "the patched line is not in the source block exactly once"]
    core = CORE.read_text()
    got = core[core.index("class LatentStats") :].rstrip("\n")
    miss += [] if got == want.replace(*PATCH) else ["the staged class differs from the source block beyond the patch"]
    miss += [] if "\nEPS = 0.01\n" in core and "EPS = 0.01" in lines[5] else ["EPS line differs"]
    return miss


# ---- our path ---------------------------------------------------------------------------------------------------------


@torch.no_grad()
def window_stats(U: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
    """(B, k) min and max over tokens of cos(h_t, u) for each window of h (B, T, d) and each row u of U (k, d): the
    code's `li / li.norm(...)` and `ux @ li.T`, batched over windows."""
    hn = h / h.norm(dim=-1, keepdim=True)
    sims = torch.matmul(U, hn.transpose(-1, -2))  # (B, k, T)
    return sims.amin(-1), sims.amax(-1)


@dataclass
class Calibration:
    lo1: np.ndarray  # pass 1 ranges (K,)
    hi1: np.ndarray
    fpr_flags: np.ndarray  # (v,) the last v windows against pass 1
    fpr: float
    lo2: np.ndarray  # pass 2: installed
    hi2: np.ndarray


def ranges(mins: np.ndarray, maxs: np.ndarray, cut: int) -> tuple[np.ndarray, np.ndarray]:
    """Ranges over windows [:-cut], the code's slice: cut = 0 is empty, giving (+inf, -inf) as its untouched arrays."""
    m, M = mins[:-cut], maxs[:-cut]
    if len(m) == 0:
        return np.full(mins.shape[1], np.inf), np.full(mins.shape[1], -np.inf)
    return m.min(0), M.max(0)


def flagged(mins: np.ndarray, maxs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(n,) any direction with s_min < c_min - EPS or s_max > c_max + EPS (a value at c + EPS is not flagged)."""
    return np.asarray(((mins < lo - EPS) | (maxs > hi + EPS)).any(1), dtype=bool)


def calibrate(mins: np.ndarray, maxs: np.ndarray) -> Calibration:
    assert mins.dtype == maxs.dtype == np.float64 and mins.shape == maxs.shape and mins.ndim == 2
    assert np.isfinite(mins).all() and np.isfinite(maxs).all(), "a cosine is not finite (a zero hidden state)"
    n = len(mins)
    v = int(n * 0.1)  # the code's expression
    lo1, hi1 = ranges(mins, maxs, v)
    f = flagged(mins[-v:], maxs[-v:], lo1, hi1) if v > 0 else np.zeros(0, dtype=bool)
    fpr = float(f.sum()) / min(n, v) if v > 0 else 0.0
    lo2, hi2 = ranges(mins, maxs, 1)
    return Calibration(lo1, hi1, f, fpr, lo2, hi2)


# ---- the staged class beside it ---------------------------------------------------------------------------------------


def reference(ww: Any, us: list[Any], cal: Sequence[Tensor], test: Sequence[Tensor]) -> dict[str, Any]:
    """The staged `LatentStats` on latents (T, L + 1, d), one window per `register`, then `set_threshold(0.0)` (the
    code's `adjust_threshold(threshold_percentile=0.0)`) and read-only registers of the test windows."""
    desc = [[f"{ix}.{i}" for i in range(len(u))] for ix, u in enumerate(us)]
    ls = ww.LatentStats(max_layers=33, us=us, us_desc=desc)
    for i, lat in enumerate(cal):
        ls.register(lat, str(i))
    fpr, trig = ls.set_threshold(0.0, return_details=True)
    test_flags = [bool(ls.register(lat, "", read_only=True)) for lat in test]
    cols = [(ix, i) for ix, u in enumerate(us) for i in range(len(u))]
    return {"fpr": float(fpr), "fpr_flags": np.asarray(trig, dtype=bool), "test_flags": np.asarray(test_flags, dtype=bool),
            "lo2": np.array([ls.minima[ix][i] for ix, i in cols]), "hi2": np.array([ls.maxima[ix][i] for ix, i in cols])}  # fmt: skip


def ours(us: list[Any], cal: Tensor, test: Tensor) -> dict[str, Any]:
    """Our path on stacked latents (n, T, L + 1, d): the same columns in the same order as `reference`."""

    def stats(lat: Tensor) -> tuple[np.ndarray, np.ndarray]:
        got = [window_stats(u, lat[:, :, ix]) for ix, u in enumerate(us) if len(u)]
        mi = torch.cat([g[0] for g in got], 1) if got else lat.new_zeros(len(lat), 0)
        mx = torch.cat([g[1] for g in got], 1) if got else lat.new_zeros(len(lat), 0)
        return mi.double().numpy(), mx.double().numpy()

    c = calibrate(*stats(cal))
    tmi, tmx = stats(test)
    return {"fpr": c.fpr, "fpr_flags": c.fpr_flags, "test_flags": flagged(tmi, tmx, c.lo2, c.hi2), "lo2": c.lo2, "hi2": c.hi2}


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Every field exactly equal, or its name and the largest difference."""
    out = [] if a["fpr"] == b["fpr"] else [f"fpr {a['fpr']} vs {b['fpr']}"]
    for k in ("fpr_flags", "test_flags", "lo2", "hi2"):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or not np.array_equal(x, y):
            gap = float(np.abs(x.astype(float) - y.astype(float)).max()) if x.shape == y.shape and x.size else "shape"
            out.append(f"{k} differ ({gap})")
    return out


# ---- the known-range test (appendix C1) --------------------------------------------------------------------------------


def unit_with(cos: Sequence[float], d: int) -> Tensor:
    """A float64 vector whose first entries are `cos` and whose computed norm is exactly 1.0, so that the code's
    normalisation leaves it unchanged and its cosine with e_i is cos[i] exactly. Entry d - 1 is set to make up the
    norm, stepped by one ulp at a time until the computed norm is 1.0 (asserted)."""
    v = torch.zeros(d, dtype=torch.float64)
    v[: len(cos)] = torch.tensor(list(cos), dtype=torch.float64)
    rest = 1.0 - float((v * v).sum())
    assert rest >= 0, cos
    s = torch.tensor(math.sqrt(rest), dtype=torch.float64)
    for k in range(200):
        for sign in (1.0, -1.0):
            t = s
            for _ in range(k):
                t = torch.nextafter(t, torch.tensor(sign * 2.0, dtype=torch.float64))
            v[d - 1] = t
            if float(v.norm()) == 1.0 and float(v[None, None].norm(dim=-1)) == 1.0:
                return v
    raise AssertionError(f"no exact unit vector for {cos}")


@dataclass(frozen=True)
class Planted:
    """The known-range test's latents and what each must give, by construction.
    Layout (ragged): hidden index 1 holds directions e_0 and e_1, index 3 holds e_2, indices 0 and 2 none.
    Calibration: n windows of T tokens, values multiples of 1/64 in [-0.25, 0.25], window 0 holding -0.25 and 0.25 on
    every direction. Window n - 2 holds c_min - 0.0101 on e_1 and window n - 1 holds c_max + 0.0101 on e_0, so pass 1
    (ranges from windows 0..n-3) flags both: FPR 1. Pass 2 (windows 0..n-2) keeps e_1's lower value.
    Test probes, one direction moved at a time, the others mid-range: inside; at c_max + 0.01 and at c_min - 0.01
    (not flagged: the code keeps min - EPS <= x <= max + EPS); at c_max + 0.0101 and c_min - 0.0101 (flagged)."""

    us: list[Any]
    cal: Tensor  # (n, T, 4, d) float64
    test: Tensor
    cal_cos: list[list[list[float]]]  # [window][token][direction] the planted cosines, columns in us order
    test_cos: list[list[list[float]]]
    want: dict[str, Any]


def planted(n: int = 20, T: int = 3, d: int = 8, seed: int = 0) -> Planted:
    assert n == 20, "the planted FPR flags assume v = int(0.1 n) = 2"
    e = torch.eye(d, dtype=torch.float64)
    us: list[Any] = [[], e[[0, 1]], [], e[[2]]]
    gen = torch.Generator().manual_seed(seed)
    cal_cos = [[[float(x) / 64 for x in torch.randint(-16, 17, (3,), generator=gen)] for _ in range(T)] for _ in range(n)]
    for j in range(3):
        cal_cos[0][0][j], cal_cos[0][1][j] = -0.25, 0.25
        for w in (n - 2, n - 1):
            for t in range(T):
                cal_cos[w][t][j] = 0.0
    lo, hi = -0.25, 0.25
    cal_cos[n - 2][1][1] = lo - 0.0101
    cal_cos[n - 1][2][0] = hi + 0.0101
    lo2 = [lo, lo - 0.0101, lo]
    probes = [("inside", None, 0.0, False)]
    for j in range(3):
        probes += [(f"max_margin_{j}", j, hi + EPS, False), (f"over_max_{j}", j, hi + 0.0101, True),
                   (f"min_margin_{j}", j, lo2[j] - EPS, False), (f"under_min_{j}", j, lo2[j] - 0.0101, True)]  # fmt: skip
    test_cos = []
    for _, j, x, _ in probes:
        rows = [[0.0, 0.0, 0.0] for _ in range(T)]
        if j is not None:
            rows[1][j] = x
        test_cos.append(rows)

    def latents(cos: list[list[list[float]]]) -> Tensor:
        out = torch.zeros(len(cos), T, 4, d, dtype=torch.float64)
        for w, win in enumerate(cos):
            for t, row in enumerate(win):
                out[w, t, 1] = unit_with([row[0], row[1]], d)
                out[w, t, 3] = unit_with([0.0, 0.0, row[2]], d)
        return out

    want = {"fpr": 1.0, "fpr_flags": np.array([True, True]), "test_flags": np.array([p[3] for p in probes]),
            "lo2": np.array(lo2), "hi2": np.array([hi, hi, hi]), "probes": [p[0] for p in probes]}  # fmt: skip
    return Planted(us, latents(cal_cos), latents(test_cos), cal_cos, test_cos, want)


def known_range(ww: Any) -> dict[str, Any]:
    """Appendix C1: our path and the staged class on the planted latents, each against the planted answers and against
    each other, exactly."""
    p = planted()
    ref = reference(ww, p.us, list(p.cal), list(p.test))
    got = ours(p.us, p.cal, p.test)
    misses = [f"ours vs planted: {m}" for m in compare(got, p.want)] + [f"class vs planted: {m}" for m in compare(ref, p.want)]
    misses += [f"ours vs class: {m}" for m in compare(got, ref)]
    return {"probes": p.want["probes"], "test_flags": got["test_flags"].tolist(), "fpr": got["fpr"], "lo2": got["lo2"].tolist(),
            "hi2": got["hi2"].tolist(), "misses": misses, "pass": not misses}  # fmt: skip
