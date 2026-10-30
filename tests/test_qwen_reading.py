"""CPU checks of E57's rules (commute.qwen_reading, experiments/qwen_reading): the route-split shares and their status,
the teeth verdict, the KL scale search, the gated-on and split-half rules, the site-stratified overlap, the filed
section 5 rules against every row of the plan's edge-case table, effective rank, and the row builders."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from commute import qwen_reading as qr
from commute.qwen_vpd import SITES

PRED = Path(__file__).resolve().parents[1] / "experiments" / "qwen_reading" / "results" / "predictions.json"
NAN = float("nan")


def cells(K00=1.4, K01=0.2, K10=1.3, K11=0.1, K0=0.07):
    return {"K00": K00, "K01": K01, "K10": K10, "K11": K11, "K0": K0}

# ----------------------------------------------------------------------------- row 1


def test_shares_values_and_do_nothing():
    s = qr.shares(cells(), 1)
    assert s["status"] == "scored"
    assert s["regating"] == pytest.approx((1.4 - 0.2) / (1.4 - 0.1))
    assert s["direct"] == pytest.approx((1.4 - 1.3) / (1.4 - 0.1))
    dn = qr.shares(cells(K10=1.4, K11=0.2), 1)  # do-nothing: K10 = K00, K11 = K01
    assert dn["regating"] == 1.0 and dn["direct"] == 0.0


@pytest.mark.parametrize("k,status", [
    (cells(K00=0.07), "unscored"),  # K00 - K0 = 0
    (cells(K00=0.05), "unscored"),  # K00 - K0 < 0 (then K00 - K11 < 0 too)
    (cells(K11=0.8), "unscored"),  # K11 - K0 > 0.5 (K00 - K0)
    (cells(K00=0.15, K01=0.1, K10=0.14, K11=0.11), "unscored"),  # denominator 0.04 < floor 0.05
    (cells(K01=NAN), "nan"),
    ({"K00": 1.0}, "nan"),
])
def test_shares_status_edges(k, status):
    assert qr.shares(k, 1)["status"] == status


def test_shares_half_line_inclusive():
    # K11 - K0 exactly 0.5 (K00 - K0): scored (the rule is strict >)
    assert qr.shares(cells(K00=1.07, K11=0.57, K01=0.3, K10=1.0), 1)["status"] == "scored"


def rows_of(n, f):
    return {k: [f(k, i) for i in range(n)] for k in ("K0", "K00", "K01", "K10", "K11")}


def test_teeth_verdicts():
    n = 64
    rng = np.random.default_rng(0)
    noise = {k: rng.normal(0, 0.002, n) for k in ("K0", "K00", "K01", "K10", "K11")}
    base = {"K0": 0.07, "K00": 1.4, "K10": 1.3, "K11": 0.1}

    def make(k01):
        return {k: [(k01 if k == "K01" else base[k]) + noise[k][i] for i in range(n)] for k in noise}

    e40, low, high = make(0.15), make(1.2), make(0.3)  # shares ~0.96, ~0.15, ~0.85
    draws = qr.resample(n, 200)
    assert qr.teeth(e40, {"a": {"rows": low, "matched": True}, "b": {"rows": low, "matched": True}},
                    draws, 1)["verdict"] == "teeth"
    assert qr.teeth(e40, {"a": {"rows": low, "matched": True}, "b": {"rows": high, "matched": True}},
                    draws, 1)["verdict"] == "no teeth"
    assert qr.teeth(e40, {"a": {"rows": low, "matched": False}, "b": {"rows": low, "matched": True}},
                    draws, 1)["verdict"] == "weak"
    mid = make(0.49)  # E40 at share 0.7 and a control at the same share: the gap's interval does not clear 0
    assert qr.teeth(mid, {"a": {"rows": low, "matched": True}, "b": {"rows": mid, "matched": True}},
                    draws, 1)["verdict"] == "weak"


def test_boot_share_covers_the_share():
    r = rows_of(32, lambda k, i: cells()[k] * (1 + 0.01 * math.sin(i)))
    lo, hi = qr.boot_share(r, qr.resample(32, 500), 1)
    s = qr.shares({k: sum(v) for k, v in r.items()}, 32)["regating"]
    assert lo <= s <= hi


def test_scale_search():
    f = lambda b: 0.003 * b**2
    out = qr.scale_search(f, 0.6, 1.5, 6)
    assert out["matched"] and len(out["evals"]) <= 6 and qr.within(out["kl"], 0.6, 1.5)
    none = qr.scale_search(f, 0.0, 1.5, 6)
    assert none["evals"] == [] and none["b"] == 1.0 and math.isnan(none["kl"]) and not none["matched"]
    bad = qr.scale_search(lambda b: NAN if b >= 1 else 0.6, 0.6, 1.5, 6)  # NaN counts as too large
    assert bad["matched"] and bad["b"] < 1
    never = qr.scale_search(lambda b: 1e-9, 0.6, 1.5, 3)  # flat: unmatched, the closest kept
    assert not never["matched"] and len(never["evals"]) == 3

# ----------------------------------------------------------------------------- rows 2 and 3


def test_gated_is_exact_at_one_percent():
    assert qr.gated(torch.tensor([1, 0, 2]), 100).tolist() == [True, False, True]
    assert qr.gated(torch.tensor([1]), 101).tolist() == [False]
    with pytest.raises(AssertionError):
        qr.gated(torch.tensor([1]), 0)


def test_split_floor_and_min_size():
    a, b = torch.tensor([1, 1, 0, 0], dtype=torch.bool), torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    assert qr.split_floor(a, b) == 2 and qr.split_floor(b, a) == 2
    assert qr.min_size(0) == 20 and qr.min_size(3) == 20 and qr.min_size(5) == 25


def test_quantile():
    d = np.arange(1, 1001)
    assert qr.quantile(d, 0.95) == 950 and qr.quantile(d, 0.99) == 990


def layout(n_sites=4, per=50):
    site_of = torch.arange(n_sites).repeat_interleave(per)
    return site_of, n_sites * per


ALL = torch.ones(200, dtype=torch.bool)  # the pool for layout(): every live subcomponent


def test_overlap_null_and_shared_set():
    site_of, n = layout()
    g = torch.Generator().manual_seed(0)
    X = torch.rand(n, generator=g) < 0.3
    S = torch.rand(n, generator=g) < 0.2
    ov = qr.overlap(S, X, site_of, 4, ALL)
    want = sum(int(S[site_of == i].sum()) * int(X[site_of == i].sum()) / 50 for i in range(4))
    assert ov["mean"] == pytest.approx(want, rel=0.05)
    same = qr.overlap(X, X, site_of, 4, ALL)  # the seed-pair control's extreme: S = X
    assert qr.above_99(same) and same["e"] > 3
    assert qr.overlap(S, X, site_of, 4, ALL) == ov  # seeded


@pytest.mark.parametrize("make", [
    lambda n: (torch.zeros(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool)),  # S empty
    lambda n: (torch.ones(n, dtype=torch.bool), torch.rand(n) < 0.5),  # S fills every live slot
    lambda n: (torch.rand(n) < 0.5, torch.zeros(n, dtype=torch.bool)),  # X empty
])
def test_overlap_sd_zero_is_undefined(make):
    site_of, n = layout()
    S, X = make(n)
    ov = qr.overlap(S, X, site_of, 4, ALL)
    assert ov["sd"] == 0 and math.isnan(ov["e"])
    assert not qr.above_99(ov) and not qr.at_most_95(ov)
    assert qr.prompt_row(1, ov, 0, 100) == qr.prompt_row(2, ov, 0, 100) == "unscored"
    assert qr.think_row(ov, 0, [50, 50, 50]) == "unscored"


def ov(o, size_S=100, sd=2.0, q95=10.0, q99=12.0):
    return {"o": o, "size_S": size_S, "sd": sd, "e": (o - 8) / sd, "q95": q95, "q99": q99}


def test_prompt_row_edges():
    assert qr.prompt_row(1, ov(13), 25, 100) == "holds"
    assert qr.prompt_row(2, ov(13), 25, 100) == "misses"
    assert qr.prompt_row(1, ov(11), 25, 100) == qr.prompt_row(2, ov(11), 25, 100) == "misses"  # between q95, q99
    assert qr.prompt_row(1, ov(12), 25, 100) == "misses"  # strictly above q99
    assert qr.prompt_row(2, ov(10), 25, 100) == "holds"  # at q95: at or below
    assert qr.prompt_row(1, ov(13, size_S=24), 25, 100) == "unscored"  # |S| under the floor: never 'at chance'
    assert qr.prompt_row(2, ov(3, size_S=24), 25, 100) == "unscored"
    assert qr.prompt_row(1, ov(13), 25, 24) == "unscored"  # |S_prompt| under the floor
    assert qr.prompt_row(1, None, 25, 100) == "unscored"  # missing


def test_think_row_edges():
    assert qr.think_row(ov(10), 25, [30, 30, 30]) == "holds"
    assert qr.think_row(ov(11), 25, [30, 30, 30]) == "misses"  # above q95 for anchor 2
    assert qr.think_row(ov(5), 25, [30, 19, 30]) == "unscored"  # |L| < 20
    assert qr.think_row(ov(5, size_S=10), 25, [30, 30, 30]) == "unscored"
    assert qr.think_row(None, 25, [30, 30, 30]) == "unscored"


def test_validated_edges():
    assert qr.validated([5.0, 4.0], [0.5, 1.0], 0.5) == {"validated": True, "gap": 3.75, "bar": 1.0,
                                                         "direction": "as filed"}
    assert qr.validated([0.5, 1.0], [5.0, 4.0], 0.5)["direction"] == "reversed"
    assert not qr.validated([5.0, None], [0.5, 1.0], 0.5)["validated"]  # an anchor seed unscored
    assert not qr.validated([5.0, NAN], [0.5, 1.0], 0.5)["validated"]
    assert not qr.validated([5.0, 4.0], [0.5, 1.0], NAN)["validated"]
    assert not qr.validated([5.0, 1.0], [0.5, 1.0], 0.1)["validated"]  # seed gap 4 > mean gap 2.25
    assert not qr.validated([5.0, 4.0], [0.5, 1.0], 2.0)["validated"]  # 2 se = 4 > 3.75
    assert not qr.validated([2.0, 2.0], [1.0, 1.0], 0.5)["validated"]  # gap equals the bar: strict

# ----------------------------------------------------------------------------- row 4


def test_effective_rank():
    assert qr.effective_rank(torch.ones(7)) == pytest.approx(7)
    assert qr.effective_rank(torch.tensor([3.0, 0, 0])) == pytest.approx(1)
    assert math.isnan(qr.effective_rank(torch.zeros(4)))


def test_placement():
    pe = torch.tensor([1.0, 3.0, 0.0, 4.0])
    assert qr.placement(pe, torch.tensor([True, False, True, True])) == pytest.approx(5 / 8)
    assert math.isnan(qr.placement(torch.zeros(3), torch.ones(3, dtype=torch.bool)))

# ----------------------------------------------------------------------------- rows


def test_unpack_and_contexts():
    P = [[100, 1, 2], [100, 3], [100, 4, 5], [100, 6]]
    R = [[7, 8, 9, 99], [10, 99], [11, 12, 13, 14], [15, 99]]  # the third never ends (truncated, no end token)
    flat = [t for p, r in zip(P, R, strict=True) for t in p + r]
    seqs = qr.unpack(flat, P)
    assert seqs == [P[0] + R[0], P[1] + R[1], P[2] + R[2]]  # the last one dropped: nothing marks its end
    assert qr.unpack(flat[:-1], P[:2]) == [P[0] + R[0]]
    with pytest.raises(AssertionError):
        qr.unpack([0, *flat], P)
    m = 50
    plain, think = [100, 1, m, 7, 8], [100, 1, 20, 21, 22, 23, m, 9]
    L = qr.length_matched(plain, think, [plain, [100, 3, m, 4], [100, 5, m, 6]], 0, m)
    assert qr.after(L, m) == qr.after(think, m) and L[-len(plain):] == plain and len(L) == 4 + len(plain)
    assert L[:4] == qr.filler([plain, [100, 3, m, 4], [100, 5, m, 6]], 0, 4)
    assert qr.length_matched(think, plain, [plain, think], 0, m) == list(think)  # no room: no filler
    assert qr.after([1, 2], m) == 2 and qr.filler([[1], [2, 3]], 0, 3) == [3, 2, 3]


def test_counts_on_a_layout():
    idx = {s: torch.tensor([0, 2]) for s in SITES}
    lay = qr.Layout(idx)
    assert lay.n == 2 * len(SITES) and int(lay.kind_mask("q").sum()) == 2 * 28
    c = qr.Counts(lay, "cpu", n_rows=2)
    g = {s: torch.tensor([[[0.5, 0.9, 0.0], [0.02, 0.9, 0.005]]]) for s in SITES}  # [1, T=2, C=3]
    c.add_row(g, row=1)
    c.add_row(g, keep=torch.tensor([True, False]))
    assert c.n_pos == 3
    assert c.active[:2].tolist() == [3, 0] and c.gsum[0].item() == pytest.approx(1.02)
    assert c.rows is not None and c.rows[1, :2].tolist() == [2, 0] and c.rows[0].sum() == 0
    assert c.on()[:2].tolist() == [True, False]


def test_predictions_name_the_code_checks():
    p = json.loads(PRED.read_text())
    assert "status" not in p and len(p["edge_cases"]) == 9
    assert {"A_W0", "A_E40", "A_do_nothing", "A_gpu", "A_gate_counts", "B1", "B2", "C0", "C_E40",
            "C_E45"} == set(p["early"])
    assert {"prompt_overlap", "think_overlap", "validated", "domain_confound"} == set(p["rules"])


def test_shares_floor_reads_mean_nats_per_position():
    n = 128 * 256  # sums over the maths rows
    low = {k: v * n for k, v in cells(K00=0.06, K01=0.03, K10=0.05, K11=0.02, K0=0.0).items()}  # den 0.04 / position
    high = {k: v * n for k, v in cells(K00=0.08, K01=0.03, K10=0.05, K11=0.02, K0=0.0).items()}  # den 0.06
    assert qr.shares(low, n)["status"] == "unscored" and qr.shares(high, n)["status"] == "scored"
    assert qr.shares(high, n)["den_mean"] == pytest.approx(0.06)
    assert qr.shares(high, 0)["status"] == "unscored"


def test_overlap_pool_puts_a_set_at_chance_at_zero():
    """A set S drawn at random inside the pool (live, off under g0) reads e ≈ 0 against a reference X that holds
    every subcomponent on at W0; drawing from every live subcomponent instead reads it far below chance."""
    site_of = torch.arange(4).repeat_interleave(200)
    g = torch.Generator().manual_seed(1)
    on_w0 = torch.rand(800, generator=g) < 0.2
    pool = ~on_w0
    X = on_w0 | (pool & (torch.rand(800, generator=g) < 0.2))
    e_pool, e_all = [], []
    for t in range(200):
        S = pool & (torch.rand(800, generator=torch.Generator().manual_seed(100 + t)) < 0.1)
        e_pool.append(qr.overlap(S, X, site_of, 4, pool, 1000, t)["e"])
        e_all.append(qr.overlap(S, X, site_of, 4, torch.ones(800, dtype=torch.bool), 1000, t)["e"])
    assert abs(float(np.mean(e_pool))) < 0.25
    assert float(np.mean(e_all)) < -1
    with pytest.raises(AssertionError):
        qr.overlap(on_w0, X, site_of, 4, pool)


def _lab_module():
    import importlib.util

    p = PRED.parents[1] / "lab.py"
    spec = importlib.util.spec_from_file_location("qwen_reading_lab", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up while the file runs
    spec.loader.exec_module(mod)
    return mod


def test_weight_sets_bookkeeping():
    """Lab's weight sets on fake linears: W0 exists from the start (job 46274 died on a set made before W0 was
    registered), the W0 control's FT is a separate copy, use copies in place, drop returns to W0."""
    from types import SimpleNamespace

    Weights = _lab_module().Weights
    w0 = {"a": torch.arange(6.0).reshape(2, 3), "b": torch.ones(4)}
    lins = {s: SimpleNamespace(weight=SimpleNamespace(data=t.clone())) for s, t in w0.items()}
    data = {s: lin.weight.data for s, lin in lins.items()}  # the model's own tensors: never replaced
    W = Weights(lins)
    W.put("FT", None)  # the first call of Stage A's controls, before any use
    assert W.sets["FT"]["a"] is not W.sets["W0"]["a"] and torch.equal(W.sets["FT"]["a"], w0["a"])
    W.use("FT")
    assert W.now == "FT" and all(torch.equal(data[s], w0[s]) for s in w0)
    W.put("FT", {s: torch.ones_like(t) for s, t in w0.items()}, 2.0)  # replacing the set the model holds
    assert W.now == "W0"
    W.use("FT")
    assert torch.equal(data["a"], w0["a"] + 2) and lins["a"].weight.data is data["a"]
    W.put("state", {s: -torch.ones_like(t) for s, t in w0.items()})  # built from W0, not from the FT held now
    assert torch.equal(W.sets["state"]["b"], w0["b"] - 1)
    W.use("state")
    W.drop("state")
    assert W.now == "W0" and "state" not in W.sets and torch.equal(data["b"], w0["b"])
    W.drop("never-made")
    with pytest.raises(AssertionError):
        W.drop("W0")
