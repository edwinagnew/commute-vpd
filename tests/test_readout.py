"""E59's filed rules (src/commute/readout.py) on hand-made numbers, with the edge cases predictions.json names: a key
with no window, even window counts, ties at every bound, no block before the last, a gain under the floor, a nan or
missing state; and the ablation's projection and random directions on small tensors (no model)."""

from __future__ import annotations

import math

import torch

from commute import readout as RO

SEEDS, OWN = ["s0", "s1", "s2", "s3", "s4"], ["s0", "s1"]


def test_key_medians_lower_median_and_empty_key() -> None:
    med = RO.key_medians([5, 1, 9, 4, 7], [1, 1, 1, 2, 2], [1, 2, 3])
    assert med == {1: 5, 2: 4, 3: None}  # odd count: middle; even count [4, 7]: lower; no window: None


def test_possession_fraction_inclusive_and_empty() -> None:
    med = {k: (20 if k < 24 else 21) for k in range(32)}  # 24 of 32 at exactly rank 20
    assert RO.possession(med, 20, 24, 32) == {"count": 24, "covered": 32, "possessed": True}
    med[0] = 21
    assert not RO.possession(med, 20, 24, 32)["possessed"]
    fit = {k: (3 if k < 21 else 50) for k in range(28)} | {k: None for k in range(28, 32)}  # 21 of 28 covered
    assert RO.possession(fit, 20, 24, 32)["possessed"]
    fit[0] = 50
    assert not RO.possession(fit, 20, 24, 32)["possessed"]
    assert RO.possession({0: None, 1: None}, 20, 24, 32) == {"count": 0, "covered": 0, "possessed": False}


def test_earliest_ignores_last_block() -> None:
    assert RO.earliest([False, True, True, True]) == 1  # ties across blocks: the earliest
    assert RO.earliest([False, False, False, True]) is None
    assert RO.earliest([True, False, False, False]) == 0


def test_p1_readings() -> None:
    assert RO.p1([False, True, False, True], [False] * 4)["status"] == "pass"
    null = RO.p1([False, False, False, True], [False] * 4)
    assert null["status"] == "fail" and null["reading"].startswith("null")
    cip = RO.p1([True, True, True, True], [False, False, False, True])
    assert cip["status"] == "fail" and cip["reading"] == "cipher possessed at blocks [3]"
    assert RO.p1([False] * 4, [False] * 4)["reading"] == "elicit possessed at no block"


def test_p3_bounds() -> None:
    e = {k: (2 if k < 12 else 20 if k < 24 else 1) for k in range(32)}
    c = {k: (101 if k < 24 else 100) for k in range(32)}
    assert RO.p3(e, c, (2, 20), 24, 100) == {"elicit_in_band": 24, "cipher_above": 24, "status": "pass"}
    e[0], c[0] = 21, 100
    out = RO.p3(e, c, (2, 20), 24, 100)
    assert (out["elicit_in_band"], out["cipher_above"], out["status"]) == (23, 23, "fail")


def test_do_nothing_bound_and_nan() -> None:
    assert RO.do_nothing({"a": 0.05, "b": -0.05}, 0.05)["pass"]
    assert not RO.do_nothing({"a": 0.0501}, 0.05)["pass"]
    assert not RO.do_nothing({"a": math.nan, "b": 0.0}, 0.05)["pass"]
    assert not RO.do_nothing({"a": None}, 0.05)["pass"]
    assert not RO.do_nothing({}, 0.05)["pass"]


def shares(elicit: list[float], cipher: list[float], own: list[float]) -> dict[str, float | None]:
    A: dict[str, float | None] = {f"elicit_{s}": v for s, v in zip(SEEDS, elicit, strict=True)}
    A |= {f"cipher_{s}": v for s, v in zip(SEEDS, cipher, strict=True)}
    return A | {f"cipher_own_{s}": v for s, v in zip(OWN, own, strict=True)}


def test_p2_pass_fail_and_bounds() -> None:
    ok = shares([0.8, 0.85, 0.82, 0.8, 0.84], [0.3, 0.32, 0.31, 0.33, 0.3], [0.35, 0.3])
    assert RO.p2(ok, SEEDS, OWN, 0.2, 0.1, None)["status"] == "pass"
    edge = shares([0.8] * 5, [0.3] * 5, [0.4, 0.3])  # own-step difference exactly 0.1: fails (strict)
    assert RO.p2(edge, SEEDS, OWN, 0.2, 0.1, None)["status"] == "fail"
    gap = shares([0.75] * 5, [0.5] * 5, [0.5, 0.5])  # gap exactly at min_gap (0.25, exact in binary), range 0: separates
    assert RO.p2(gap, SEEDS, OWN, 0.25, 0.1, None)["status"] == "pass"
    spread = shares([0.7, 0.9, 0.8, 0.8, 0.8], [0.2, 0.4, 0.3, 0.3, 0.3], [0.25, 0.4])  # gap 0.5 > range 0.2
    assert RO.p2(spread, SEEDS, OWN, 0.2, 0.1, None)["status"] == "pass"
    wide = shares([0.4, 1.0, 0.7, 0.7, 0.7], [0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2])  # gap 0.5 < range 0.6
    assert RO.p2(wide, SEEDS, OWN, 0.2, 0.1, None)["status"] == "fail"
    rev = shares([0.3] * 5, [0.8] * 5, [0.8, 0.8])
    assert RO.p2(rev, SEEDS, OWN, 0.2, 0.1, None)["separation"]["verdict"] == "b_higher"


def test_p2_unreadable_and_void() -> None:
    bad = shares([0.8, math.nan, 0.8, 0.8, 0.8], [0.3] * 5, [0.3, 0.3])
    assert RO.p2(bad, SEEDS, OWN, 0.2, 0.1, None)["status"] == "unreadable"
    missing = shares([0.8] * 5, [0.3] * 5, [0.3, 0.3])
    del missing["cipher_own_s1"]
    assert RO.p2(missing, SEEDS, OWN, 0.2, 0.1, None)["status"] == "unreadable"
    assert RO.p2({}, SEEDS, OWN, 0.2, 0.1, "no l*") == {"status": "void", "reason": "no l*"}


def test_branch() -> None:
    p1p, p1f = RO.p1([True, False, False, True], [False] * 4), RO.p1([False, False, False, True], [False] * 4)
    good = RO.p2(shares([0.8] * 5, [0.3] * 5, [0.3, 0.3]), SEEDS, OWN, 0.2, 0.1, None)
    assert RO.branch(p1p, good).startswith("P1 and P2 hold")
    assert RO.branch(p1f, good).startswith("P1 fails")
    assert RO.branch(p1f, RO.p2({}, SEEDS, OWN, 0.2, 0.1, "no l*")).startswith("P2 void")
    assert "P2 fails" in RO.branch(p1p, RO.p2(shares([0.3] * 5, [0.3] * 5, [0.3, 0.3]), SEEDS, OWN, 0.2, 0.1, None))


def test_remove_at_projects_out_only_the_answer_position() -> None:
    x = torch.randn(3, 5, 4)
    U = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
    y = RO.remove_at(2, U)(x)
    assert torch.allclose((y[:, 2] * U).sum(-1), torch.zeros(3), atol=1e-6)
    assert torch.equal(y[:, [0, 1, 3, 4]], x[:, [0, 1, 3, 4]])
    assert torch.allclose(RO.remove_at(2, U)(y), y, atol=1e-6)  # idempotent


def test_random_dirs_unit_and_shared_per_value() -> None:
    tgt = torch.tensor([7, 3, 7, 9])
    R = RO.random_dirs(tgt, 16, 5959)
    assert torch.allclose(R.norm(dim=1), torch.ones(4), atol=1e-6)
    assert torch.equal(R[0], R[2]) and not torch.equal(R[0], R[1])
    assert torch.equal(R, RO.random_dirs(tgt, 16, 5959))


def test_next_token_xent_skips_position_zero() -> None:
    V, T = 5, RO.WIN
    toks = torch.randint(0, V, (2, T))
    logits = torch.zeros(1, 2, T, V)
    logits[0, :, 0] = 100.0  # position 0 is left out: a huge logit there must not move the loss
    assert torch.allclose(RO.next_token_xent(logits, toks), torch.tensor([math.log(V)]))


def pair(lp: list[float], lo: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(lp, dtype=torch.float64), torch.tensor(lo, dtype=torch.float64)


def test_shares_log_odds_net_of_w0() -> None:
    w0, w0_abl = pair([-6.0, -6.0], [-6.0, -6.0]), pair([-7.0, -7.0], [-7.0, -7.0])  # W0 loses 1 logit
    st, st_abl = pair([-0.1, -0.1], [9.0, 9.0]), pair([-1.0, -1.0], [4.0, 4.0])  # state loses 5 logits
    out = RO.shares(w0, w0_abl, st, st_abl, 0.05)
    assert math.isclose(out["G_logodds"], 15.0) and math.isclose(out["lo_drop"], 5.0) and math.isclose(out["lo_drop_W0"], 1.0)
    assert math.isclose(out["A_lo"], 4.0 / 15.0) and math.isclose(out["A_lo_raw"], 5.0 / 15.0)
    assert math.isclose(out["A_nats"], 0.9 / 5.9) and math.isclose(out["nats_lost"], 0.9)


def test_shares_identity_edit_and_w0_state() -> None:
    w0 = pair([-6.0, -5.0], [-6.0, -5.0])
    out = RO.shares(w0, w0, w0, w0, 0.05)  # W0 itself (G 0) with a no-op removal
    assert math.isnan(out["A_lo"]) and math.isnan(out["A_nats"]) and out["lo_drop"] == 0.0 and out["median_drop"] == 0.0


def test_shares_zero_and_negative_gain() -> None:
    w0, w0_abl = pair([-6.0], [-6.0]), pair([-7.0], [-7.0])
    tiny = RO.shares(w0, w0_abl, pair([-5.99], [-5.96]), pair([-6.99], [-6.96]), 0.05)  # |G_logodds| 0.04 < floor
    assert math.isnan(tiny["A_lo"])
    zero = RO.shares(w0, w0_abl, w0, w0_abl, 0.05)
    assert math.isnan(zero["A_lo"]) and math.isnan(zero["A_lo_raw"])
    neg = RO.shares(w0, w0_abl, pair([-8.0], [-8.0]), pair([-10.0], [-10.0]), 0.05)  # gain -2, drop 2, W0 drop 1
    assert math.isclose(neg["A_lo"], -0.5)


def test_column_geometry() -> None:
    probes = RO.Probes(1, 3, 4)
    with torch.no_grad():
        probes.w.copy_(torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]]))
    g = RO.column_geometry(probes, 0, [0, 3])
    assert math.isclose(g["0"]["norm"], 1.0) and math.isclose(g["0"]["cos_mean"], 1 / math.sqrt(2), rel_tol=1e-6)
    assert math.isclose(g["3"]["norm"], math.sqrt(17), rel_tol=1e-6)
