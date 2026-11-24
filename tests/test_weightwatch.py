"""P2.73 (experiments/weightwatch): the staged WeightWatch `LatentStats` (one storage-only patch), our vectorised path and
a pure-Python reference written from the spec agree exactly on synthetic ragged data: both threshold passes, both tails,
and a value exactly at the margin (c + 0.01, not flagged). Also the staged core against its source text, the trigger
replay's algebra, and the filed rules on their edge cases. No model is built or loaded."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from commute.jobs import EXPERIMENTS, load_experiment

PRED = EXPERIMENTS / "weightwatch" / "results" / "predictions.json"


@pytest.fixture(scope="module")
def ww() -> ModuleType:
    return load_experiment("weightwatch", "ww_core")


@pytest.fixture(scope="module")
def dt() -> ModuleType:
    return load_experiment("weightwatch", "ww_detector")


@pytest.fixture(scope="module")
def sc() -> ModuleType:
    return load_experiment("weightwatch", "ww_score")


def pure(cal: list[list[list[float]]], test: list[list[list[float]]], eps: float = 0.01) -> dict[str, Any]:
    """The spec in plain Python, from the cosines themselves: per window and direction the min and max over tokens;
    pass 1 ranges from the first n - int(0.1 n) windows, the FPR on the rest; pass 2 from all windows but the last;
    flagged when s_min < c_min - eps or s_max > c_max + eps for any direction."""

    def mm(win: list[list[float]]) -> tuple[list[float], list[float]]:
        k = len(win[0])
        return [min(t[j] for t in win) for j in range(k)], [max(t[j] for t in win) for j in range(k)]

    def rng(ws: list[tuple[list[float], list[float]]], k: int) -> tuple[list[float], list[float]]:
        if not ws:
            return [math.inf] * k, [-math.inf] * k
        return [min(w[0][j] for w in ws) for j in range(k)], [max(w[1][j] for w in ws) for j in range(k)]

    def flag(st: tuple[list[float], list[float]], lo: list[float], hi: list[float]) -> bool:
        return any(st[0][j] < lo[j] - eps or st[1][j] > hi[j] + eps for j in range(len(lo)))

    stats, k = [mm(w) for w in cal], len(cal[0][0])
    n = len(stats)
    v = int(n * 0.1)
    lo1, hi1 = rng(stats[: n - v], k)
    ff = [flag(st, lo1, hi1) for st in stats[n - v :]] if v else []
    lo2, hi2 = rng(stats[: n - 1], k)
    return {"fpr": sum(ff) / v if v else 0.0, "fpr_flags": np.array(ff, dtype=bool), "test_flags": np.array([flag(mm(w), lo2, hi2) for w in test]),
            "lo2": np.array(lo2), "hi2": np.array(hi2)}  # fmt: skip


def three_way(ww: ModuleType, dt: ModuleType, us: list[Any], cal: torch.Tensor, test: torch.Tensor, cal_cos: Any, test_cos: Any) -> dict[str, Any]:
    p = pure(cal_cos, test_cos)
    ref = dt.reference(ww, us, list(cal), list(test))
    got = dt.ours(us, cal, test)
    assert dt.compare(ref, p) == [], "the staged class against the pure-Python reference"
    assert dt.compare(got, p) == [], "our path against the pure-Python reference"
    assert dt.compare(got, ref) == [], "our path against the staged class"
    return p


def test_known_range_planted(ww: ModuleType, dt: ModuleType) -> None:
    p = dt.planted()
    want = three_way(ww, dt, p.us, p.cal, p.test, p.cal_cos, p.test_cos)
    assert dt.compare(want, p.want) == []
    flags = dict(zip(p.want["probes"], p.want["test_flags"].tolist(), strict=True))
    for j in range(3):  # both tails, each at its margin and just past it
        assert (flags[f"max_margin_{j}"], flags[f"over_max_{j}"], flags[f"min_margin_{j}"], flags[f"under_min_{j}"]) == (False, True, False, True)
    assert not flags["inside"] and want["fpr"] == 1.0
    assert dt.known_range(ww)["pass"]


def ragged(n: int, n_test: int, seed: int) -> tuple[list[Any], torch.Tensor, torch.Tensor, Any, Any]:
    """Random cosines (multiples of 1/128 in [-0.4, 0.4]) on a ragged layout: index 1 holds 2 directions, index 3 one;
    test windows probe each direction at, past and inside the calibration's extremes (both tails)."""
    dt = load_experiment("weightwatch", "ww_detector")
    gen = torch.Generator().manual_seed(seed)
    e = torch.eye(8, dtype=torch.float64)
    us: list[Any] = [[], e[[0, 1]], [], e[[2]]]
    cal = [[[float(x) / 128 for x in torch.randint(-51, 52, (3,), generator=gen)] for _ in range(4)] for _ in range(n)]
    lo = [min(t[j] for w in cal[: n - 1] for t in w) for j in range(3)]
    hi = [max(t[j] for w in cal[: n - 1] for t in w) for j in range(3)]
    test = []
    for i in range(n_test):
        j, kind = i % 3, (i // 3) % 5
        x = [hi[j] + 0.01, hi[j] + 0.0101, lo[j] - 0.01, lo[j] - 0.0101, (lo[j] + hi[j]) / 2][kind]
        test.append([[x if (jj == j and t == 2) else (lo[jj] + hi[jj]) / 2 for jj in range(3)] for t in range(4)])

    def lat(cos: Any) -> torch.Tensor:
        out = torch.zeros(len(cos), 4, 4, 8, dtype=torch.float64)
        for w, win in enumerate(cos):
            for t, r in enumerate(win):
                out[w, t, 1], out[w, t, 3] = dt.unit_with([r[0], r[1]], 8), dt.unit_with([0.0, 0.0, r[2]], 8)
        return out

    return us, lat(cal), lat(test), cal, test


@pytest.mark.parametrize(("n", "seed"), [(50, 1), (37, 2), (9, 3)])  # v = 5, 3 and 0 (pass 1 empty, FPR 0, as the code)
def test_random_ragged(ww: ModuleType, dt: ModuleType, n: int, seed: int) -> None:
    us, cal, test, cc, tc = ragged(n, 30, seed)
    p = three_way(ww, dt, us, cal, test, cc, tc)
    assert p["test_flags"].any() and not p["test_flags"].all()


def test_core_is_verbatim(dt: ModuleType) -> None:
    s = json.loads(PRED.read_text())["settings"]
    src = EXPERIMENTS.parent / s["source"]
    if not src.is_file():
        pytest.skip("the WeightWatch clone is not on this machine")
    assert dt.check_core(src, s["source_sha256"], s["core_sha256"]) == []


def test_unpatched_store_refuses_ragged() -> None:
    with pytest.raises(ValueError):  # the reason for the one patch: NumPy >= 1.24 refuses a ragged list
        np.array([[], [(0.1, 0.2)], []])


def test_calibrate_edges(dt: ModuleType) -> None:
    c = dt.calibrate(np.zeros((0, 3)), np.zeros((0, 3)))
    assert c.fpr == 0.0 and np.isinf(c.lo2).all() and len(c.fpr_flags) == 0
    c = dt.calibrate(np.zeros((20, 0)), np.zeros((20, 0)))
    assert c.fpr == 0.0 and not c.fpr_flags.any()
    with pytest.raises(AssertionError):
        dt.calibrate(np.full((20, 1), np.nan), np.zeros((20, 1)))


def test_trigger_replay_algebra() -> None:
    """trigger_cos from the five scalars equals the cosine of a + (act' - act) c_old + g act' u computed directly."""
    ps = load_experiment("weightwatch", "ww_passes")
    gen = torch.Generator().manual_seed(4)
    a, c = torch.randn(6, 5, 16, generator=gen, dtype=torch.float64), torch.randn(16, generator=gen, dtype=torch.float64)
    u = torch.randn(16, generator=gen, dtype=torch.float64)
    u = u / u.norm()
    act, act2 = torch.rand(6, 5, generator=gen, dtype=torch.float64), torch.rand(6, 5, generator=gen, dtype=torch.float64) * 3
    r = {"au": a @ u, "ac": a @ c, "aa": (a * a).sum(-1), "act": act, "act2": act2, "cu": c @ u, "cc": c @ c}
    for g in (0.0, 1.0, 64.0):
        f = a + ((act2 - act)[..., None] * c) + g * act2[..., None] * u
        cos = (f @ u) / f.norm(dim=-1)
        mi, mx = ps.trigger_cos(r, g)
        assert np.allclose(mi[:, 0], cos.amin(1).numpy(), rtol=0, atol=1e-12) and np.allclose(mx[:, 0], cos.amax(1).numpy(), rtol=0, atol=1e-12)


# ---- the filed rules on their edge cases ------------------------------------------------------------------------------


def test_column_rule(sc: ModuleType) -> None:
    S = {"x2": 2.0, "new_few": 1.2, "new_thin": 1.5}
    assert sc.column_rule(S, 0.3, 0.97, 0.95, True)["earned"]
    assert not sc.column_rule(S, 0.5, 0.97, 0.95, True)["earned"]  # gap equal to the spread: not "more than"
    assert not sc.column_rule(S, 0.6, 0.97, 0.95, True)["earned"]
    assert not sc.column_rule(S, 0.3, 0.94, 0.95, True)["earned"]
    assert sc.column_rule(S, 0.3, 0.95, 0.95, True)["earned"]  # the trigger bar is "at least"
    assert not sc.column_rule(S, 0.3, None, 0.95, True)["earned"]
    assert not sc.column_rule(S, 0.3, 0.97, 0.95, None)["earned"]
    assert not sc.column_rule(S, None, 0.97, 0.95, True)["earned"]
    r = sc.column_rule(S | {"new_thin": None}, 0.3, 0.97, 0.95, True)
    assert not r["earned"] and r["gap"] is None and r["reading"] == "related-work negative"


def test_trained_reading(sc: ModuleType) -> None:
    assert sc.trained_reading(None, [1.0], 1.0).startswith("undefined")
    assert sc.trained_reading(1.0, [], 1.0) == "no Haar draws"
    assert sc.trained_reading(1.0, [0.9, 1.0], 2.0) == "inside or below its Haar null"  # a tie with the top draw
    assert sc.trained_reading(1.5, [0.9, 1.1, None], 1.5) == "at the level the gradient geometry gives"
    assert sc.trained_reading(1.6, [0.9, 1.1], 1.5) == "above both nulls"
    assert sc.trained_reading(1.6, [0.9, 1.1], None).endswith("S_G undefined")


def test_small_rules(sc: ModuleType) -> None:
    assert sc.gaussian_check([])["pass"] is None and sc.gaussian_check([1.0])["pass"] is None
    assert sc.gaussian_check([1.0, 1.0])["pass"] and not sc.gaussian_check([2.0, 2.0])["pass"]
    assert sc.gaussian_check([0.9, 1.1, 1.0, None])["pass"]
    assert sc.rel_within(1.0, 0.0, 0.1) is None and sc.rel_within(None, 1.0, 0.1) is None
    assert sc.rel_within(1.25, 1.0, 0.25) and not sc.rel_within(1.5, 1.0, 0.25)  # at the bar passes
    assert sc.spread([1.0]) is None and sc.spread([1.0, None, 3.0]) == 2.0
    assert sc.wilson(0, 0) is None and sc.wilson(0, 100)[0] == 0.0 and sc.wilson(100, 100)[1] == 1.0


def test_svd_and_s(sc: ModuleType) -> None:
    u, s, s_all = sc.kept_svd(torch.zeros(6, 4), 0.0, 20)
    assert u.shape[1] == 0 and sc.pooled({"m": (u, s)}, {"m": torch.eye(6, dtype=torch.float64)}) is None
    dW = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    u, s, s_all = sc.kept_svd(dW, 1.0, 20)
    assert u.shape[1] == 2 and sc.beyond(s_all, 2) == 1.0 and sc.beyond(s_all, 9) == 0.0  # strict: sigma = tau is left out
    assert sc.kept_svd(torch.randn(8, 30), 0.0, 3)[0].shape[1] == 3
    C = torch.diag(torch.tensor([4.0, 1.0, 1.0, 2.0], dtype=torch.float64))
    e = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(sc.a_of(e, C), torch.tensor([2.0, 0.5, 0.5, 1.0], dtype=torch.float64))
    Q = sc.haar(5, 0)
    assert torch.allclose(Q @ Q.T, torch.eye(5, dtype=torch.float64), atol=1e-12)
    assert sc.quad_floor(e[:, 0], C, 4.0) > 0
    assert abs(sc.exact_s(torch.outer(e[:, 0], torch.ones(3, dtype=torch.float64)), 1, "m", {"m": C}) - 2.0) < 1e-12


def test_settings_load() -> None:
    """The filed settings build the frozen config (fields, sizes and the cross-checks of __post_init__) without a GPU."""
    if not PRED.is_file():
        pytest.skip("no predictions file")
    s = json.loads(PRED.read_text())["settings"]
    assert len(s["panel_tasks"]) == 12 and s["k"] == 20 and (s["calib_rows"][1] - s["calib_rows"][0]) * 4 == 50_000
    assert Path(EXPERIMENTS / "weightwatch" / "ww_core.py").is_file()
