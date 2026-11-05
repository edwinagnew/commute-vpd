"""E90's filed rules (experiments/new_vs_reuse/rules.py) on hand-made numbers and their edge cases: predictions.json's rows A1-A9,
R1-R7, the do-nothing state, nan, empty, ties. No model."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from commute import readout as RO
from commute import removal as RM
from commute import steering as ST
from commute.jobs import load_experiment

R = load_experiment("new_vs_reuse", "rules")
nan = math.nan
PRED = Path(__file__).resolve().parents[1] / "experiments" / "new_vs_reuse" / "results" / "predictions.json"
KEYS = np.arange(100, 132)


def test_a1_random_map_ties_at_zero() -> None:
    r = R.parity(KEYS, KEYS + 1000)
    assert (r["best"], r["floor"], r["admissible"], r["label"]) == (0.0, 0.0, False, "not admissible")


def test_a2_copy_is_admissible_on_unseen_keys() -> None:
    r = R.parity(KEYS, KEYS)
    assert r["acc_identity"] == 1.0 and r["acc_table"] == 0.0 and r["admissible"]


def test_a3_constant_does_not_apply_whatever_the_reading() -> None:
    for reading in (KEYS, np.full(32, 7)):
        r = R.parity(reading, np.full(32, 80))
        assert r["label"] == R.DOES_NOT_APPLY and not r["admissible"] and r["floor"] == 1.0


def test_a4_rank1_identity_beats_the_commonest_token() -> None:
    v = np.array([15] * 9 + list(range(200, 215)))
    r = R.parity(v, v)
    assert r["acc_identity"] == 1.0 and r["admissible"] and r["floor"] < 1.0


def test_a5_one_value_is_a_tie_not_admissible() -> None:
    r = R.parity(np.full(24, 15), np.full(24, 15))
    assert r["best"] == r["floor"] == 1.0 and R.matches("not admissible", r) and r["label"] == R.DOES_NOT_APPLY


def test_a6_shared_guess_with_distinct_answers_ties() -> None:
    r = R.parity(np.array([15, 15, 15, 15, 16, 16, 16, 16]), np.arange(300, 308))
    assert (r["best"], r["floor"], r["admissible"]) == (0.0, 0.0, False)


def test_a7_a8_disjoint_halves() -> None:
    fit_b, read_b = np.arange(0, 40), np.arange(40, 80)
    words = np.arange(500, 540)
    look = R.admissibility(fit_b, words, read_b, words)
    assert look["unseen"] == 1.0 and not look["admissible"]
    assert R.admissibility(fit_b, fit_b, read_b, read_b)["admissible"]


@pytest.mark.parametrize("r", [np.array([7]), np.empty(0, dtype=int)])
def test_a9_no_held_out_input(r: np.ndarray) -> None:
    out = R.parity(r, r)
    assert out["label"] == R.NO_HELD_OUT and not out["admissible"] and math.isnan(out["best"]) and R.matches(R.NO_HELD_OUT, out)


def test_held_out_equals_the_shared_lookup_on_a_parity_split() -> None:
    gen = np.random.default_rng(0)
    reading, answer = gen.integers(0, 5, 41), gen.integers(0, 4, 41)
    ours, theirs = R.parity(reading, answer), RM.lookup_accuracy(reading, answer)
    assert ours["acc_table"] == theirs["acc"] and ours["floor"] == theirs["floor"] and ours["unseen"] == theirs["unseen"]


def test_possession_needs_both_and_a_tie_reaches() -> None:
    assert R.possessed(0.75, 0.1, 1.0, 3, 4)
    assert not R.possessed(0.74, 0.1, 1.0, 3, 4)
    assert not R.possessed(0.8, 0.8, 1.0, 3, 4)
    assert not R.possessed(nan, 0.1, 1.0, 3, 4)
    assert not R.possessed(0.8, 0.1, nan, 3, 4)


def test_b_star_starts_at_one_and_stops_before_the_last() -> None:
    assert R.b_star({0: True, 1: False, 2: True, 3: True}, 1, 3) == 2
    assert R.b_star({0: True, 1: False, 2: False, 3: True}, 1, 3) is None
    assert R.b_star({}, 1, 3) is None


@pytest.mark.parametrize(("s", "c", "want"), [(0.5, 0.0, R.REUSES), (0.0, 0.5, R.NOT_THROUGH), (0.5, 0.5, R.NOT_SPECIFIC), (0.0, 0.0, R.NO_REACH)])
def test_the_two_by_two(s: float, c: float, want: str) -> None:
    assert R.verdict(True, s, c, 0.1) == want


def test_no_size_bar_and_the_verdict_prints_its_share() -> None:
    assert R.verdict(True, 0.03, 0.0, 0.01) == R.REUSES
    assert R.say(R.REUSES, 0.03) == "reuses C, 0.030 of the gain" and R.say(R.NO_REACH, 0.03) == R.NO_REACH


def test_r1_do_nothing_cannot_pass() -> None:
    """The do-nothing state through the real share: zero gain is nan, and nan is no gain whatever the control reads."""
    z = torch.zeros(4, dtype=torch.float64)
    s = RO.shares((z, z), (z - 1, z - 1), (z, z), (z - 1, z - 1), 0.05)["A_lo"]
    assert math.isnan(s)
    for c in (0.0, 0.9, nan):
        assert R.verdict(True, s, c, 0.1) == R.NO_GAIN
    assert R.r2(True, s, s, s, 0.1)["reading"] == R.MISS
    assert R.r3(True, s, s, 0.1)["reading"] == R.MISS
    assert R.r7(True, s, s, 0.1)["reading"] == R.MISS


def test_r6_ties_and_nan() -> None:
    assert R.verdict(True, 0.1, 0.0, 0.1) == R.NO_REACH
    assert R.verdict(True, 0.5, nan, 0.1) == R.NO_CALL
    assert R.verdict(True, 0.5, 0.5, nan) == R.NO_CALL
    assert R.verdict(True, 0.5, 0.0, 0.1, control_ok=False) == R.NO_CALL


def test_no_teeth_comes_first_and_a_nan_drop_has_none() -> None:
    assert R.verdict(False, nan, nan, nan) == R.NO_TEETH
    assert not R.teeth(nan, 0.1) and not R.teeth(0.5, nan) and not R.teeth(0.1, 0.1)


def test_r2_readings() -> None:
    assert R.r2(True, 0.98, 0.97, 0.99, 0.05)["reading"] == R.AS_FILED
    assert R.r2(True, 0.75, 0.5, 0.625, 0.25)["reading"] == R.AS_FILED  # a spread equal to the band is within
    assert R.r2(True, 0.98, 0.7, 0.99, 0.05)["reading"] == R.MISS  # all above, not within one band
    assert R.r2(True, 0.98, 0.01, 0.99, 0.05)["reading"] == "branch: cipher spared"
    assert R.r2(True, 0.98, 0.97, 0.01, 0.05)["reading"] == R.MISS  # the hand write spared and the cipher not
    assert R.r2(False, 0.98, 0.97, 0.99, 0.05)["reading"] == R.MISS
    assert R.r2(True, 0.98, nan, 0.99, 0.05)["reading"] == R.MISS
    assert R.r2(True, 0.98, 0.97, 0.99, nan)["reading"] == R.MISS


def test_r3_readings() -> None:
    assert R.r3(True, 0.9, 0.8, 0.05)["reading"] == R.AS_FILED
    assert R.r3(True, 0.9, 0.01, 0.05)["reading"] == "branch: hand write spared"
    assert R.r3(True, 0.01, 0.01, 0.05)["reading"] == "branch: hand write spared"
    assert R.r3(True, 0.01, 0.9, 0.05)["reading"] == R.MISS
    assert R.r3(False, 0.9, 0.8, 0.05)["reading"] == R.MISS
    assert R.r3(True, 0.9, 0.01, 0.05, control_ok=False)["reading"] == R.MISS
    assert R.r3(None, nan, nan, nan)["verdict"] == "unreadable: no readout"


def test_r7_readings() -> None:
    assert R.r7(True, 0.9, 0.8, 0.05)["reading"] == R.AS_FILED
    assert R.r7(True, 0.9, 0.01, 0.05)["reading"].startswith("branch")
    assert R.r7(True, 0.01, 0.01, 0.05)["reading"] == R.MISS  # the branch needs the state above
    assert R.r7(True, 0.01, 0.9, 0.05)["reading"] == R.MISS
    assert R.r7(True, 0.9, nan, 0.05)["reading"] == R.MISS


def test_a_miss_is_a_gap_and_stops_nothing() -> None:
    ok = {"reading": R.AS_FILED}
    assert R.gaps({"R2": ok, "R3": ok, "R7": ok}) == []
    assert R.gaps({"R2": {"reading": R.MISS}, "R3": {"reading": "branch: no readout"}, "R7": ok}) == ["R2"]
    assert R.gaps({"I2": R.gap(R.NO_VERSION), "I1": R.gap(R.NO_C), "I3": ok}) == ["I2", "I1"]


IS, ARE, WAS = 310, 1083, 369


@pytest.mark.parametrize(
    ("answers", "label"),
    [([IS, IS, ARE, IS, WAS, IS, IS, ARE, IS, WAS, IS, ARE, IS, IS, WAS, IS], "admissible"),  # 3 answers over 16 keys, floor 0.625
     ([IS, IS, ARE, IS] + [IS] * 12, "not admissible"),  # one odd key in the fit half: floor 1.0, a gap by the split
     ([IS] * 16, R.DOES_NOT_APPLY),
     ([IS], R.NO_HELD_OUT)],
)
def test_i1_table_admissibility(answers: list[int], label: str) -> None:
    a = np.array(answers)
    assert R.parity(a, a)["label"] == label


def test_i3_two_sided_case_is_admissible_and_one_sided_does_not_apply() -> None:
    reading = np.array([0] * 20 + [1] * 20)
    two = R.parity(reading, reading)
    assert two["admissible"] and two["floor"] == 0.5
    assert R.parity(np.ones(40, dtype=int), np.ones(40, dtype=int))["label"] == R.DOES_NOT_APPLY


def test_the_table_must_keep_the_old_target() -> None:
    assert R.keeps_target(WAS, [IS, WAS], [IS, WAS, IS])
    assert not R.keeps_target(WAS, [IS, ARE], [IS, ARE])  # it left V
    assert not R.keeps_target(WAS, [IS, WAS], [IS, IS])  # a value, but no key answers it
    assert not R.keeps_target(WAS, [], [])
    r = R.gap(R.NO_VERSION, windows_top1_target=3)
    assert r["verdict"] == R.NO_VERSION and r["windows_top1_target"] == 3


def test_i0_do_nothing_on_the_new_rows_cannot_pass() -> None:
    z = torch.zeros(4, dtype=torch.float64)
    s = RO.shares((z, z), (z - 1, z - 1), (z, z), (z - 1, z - 1), 0.05)["A_lo"]
    assert math.isnan(s) and R.r3(True, s, 0.9, 0.1)["verdict"] == R.NO_GAIN and R.i3(True, s, 0.9, 0.1)["verdict"] == R.NO_GAIN
    assert R.r3(True, s, 0.9, 0.1)["reading"] == R.MISS and R.i3(True, s, 0.9, 0.1)["reading"] == R.MISS


def test_i3_readings() -> None:
    assert R.i3(True, 0.9, 0.8, 0.05)["reading"] == R.AS_FILED
    assert R.i3(True, 0.9, 0.01, 0.05)["reading"] == "branch: push spared, state above: reuses C"
    assert R.i3(True, 0.01, 0.01, 0.05)["reading"] == "branch: push spared, state within: does not reach"
    assert R.i3(True, 0.01, 0.9, 0.05)["reading"] == R.MISS and R.i3(True, 0.01, 0.9, 0.05)["verdict"] == R.NOT_THROUGH
    no_call = R.i3(True, 0.9, nan, 0.05, control_ok=False)
    assert no_call["verdict"] == R.NO_CALL and no_call["reading"] == R.MISS
    assert R.r3(True, 0.9, nan, 0.05, control_ok=False)["verdict"] == R.NO_CALL


def test_table_row_names_the_reason_for_an_unreadable_row() -> None:
    assert R.table_row("x", "old", "admissible", R.REUSES)["reason"] == ""
    assert R.table_row("x", "old", "admissible", R.NOT_SPECIFIC)["reason"] == R.NOT_SPECIFIC
    assert R.table_row("x", "old", R.DOES_NOT_APPLY, None) == {"task": "x", "assumed": "old", "verdict": R.DOES_NOT_APPLY, "reason": R.DOES_NOT_APPLY}
    assert R.table_row("x", "new", None, None)["verdict"] == "no C listed"


def test_seeds_mixed_empty_and_counted() -> None:
    assert R.over_seeds([R.REUSES, R.REUSES])["verdict"] == R.REUSES
    mixed = R.over_seeds([R.REUSES, R.NO_GAIN, R.NO_CALL])
    assert mixed["verdict"] == "mixed" and mixed["n_no_gain"] == 1 and mixed["n_no_call"] == 1
    assert R.over_seeds([])["verdict"] == R.NO_CALL
    assert R.all_pairs(True, [], [0.5], 0.1)["verdict"] == R.NO_CALL


def test_r4_r5_on_the_saved_numbers_shape() -> None:
    assert R.all_pairs(True, [0.10, 0.17], [0.024, 0.041], 0.0042)["verdict"] == R.NOT_SPECIFIC
    assert R.verdict(True, 0.47, -0.003, 0.24) == R.REUSES
    assert R.verdict(True, -0.003, 0.47, 0.24) == R.NOT_THROUGH


def test_every_filed_row_is_there() -> None:
    P = json.loads(PRED.read_text())
    assert set(P["admissibility"]["verdicts"]) == {f"A{i}" for i in range(1, 10)}
    assert set(P["edge_cases"]) == {f"R{i}" for i in range(1, 8)} | {f"I{i}" for i in range(4)}


class Echo:
    """A stand-in net: W0's argmax at each position is the next token (the last position: token 0), residuals are random."""

    device = "cpu"

    def __init__(self, vocab: int) -> None:
        self.net = type("M", (), {"lm_head": type("H", (), {"weight": torch.zeros(vocab, 4)})()})()
        self.vocab = vocab

    def run(self, toks: torch.Tensor, edit: object = None, capture: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        nxt = torch.cat([toks[:, 1:], torch.zeros_like(toks[:, :1])], 1)
        h = torch.randn(*toks.shape, 4)
        return torch.nn.functional.one_hot(nxt, self.vocab).float(), {s: h for s in RO.RESID} if capture else {}


def test_guess_scan_recuts_with_the_position_at_half() -> None:
    AP = load_experiment("new_vs_reuse", "apparatus")
    half, vals = 20, torch.tensor([3, 5])
    rows = torch.randint(0, 9, (40, 512))
    net = Echo(9)
    cand, raw = AP.scan_guess(net, rows, vals, half, 50, 30, 0)
    assert cand.shape[1] == RO.WIN and raw[3] > 0 and raw[5] > 0
    toks, h, lg, y = AP.keep_guess(net, cand, vals, half, 50)
    assert bool((toks[:, half + 1] == vals[y]).all()) and h.shape[:2] == (4, len(y)) and lg.shape == (len(y), 2)
    assert bool((torch.bincount(y) <= 50).all()) and bool((lg.argmax(1) == y).all())


def test_removal_edit_touches_only_the_answer_position() -> None:
    AP = load_experiment("new_vs_reuse", "apparatus")
    x, u = torch.randn(3, 5, 8), RM.random_basis(8, 2, [1])
    y = AP.removal_edit(1, 2, u)[RO.RESID[1]](x)
    assert (y[:, 2] @ u).abs().max() < 1e-5 and torch.equal(y[:, [0, 1, 3, 4]], x[:, [0, 1, 3, 4]])


def test_reads_with_alt_is_the_pairwise_difference_and_absent_is_unchanged() -> None:
    AP = load_experiment("new_vs_reuse", "apparatus")
    net = Echo(9)
    net.at = lambda _dW: nullcontext()  # type: ignore[attr-defined]
    toks = torch.randint(0, 9, (6, 41))
    tgt, alt = torch.full((6,), 3), torch.full((6,), 4)
    lp, lo = AP.reads(net, AP.Site(toks, tgt, 20))
    lp2, d = AP.reads(net, AP.Site(toks, tgt, 20, alt))
    logits = net.run(toks)[0]
    assert torch.equal(lp, lp2) and torch.equal(lo, ST.logodds_at(logits, 20, tgt))
    assert torch.allclose(d, ST.logp_at(logits, 20, tgt) - ST.logp_at(logits, 20, alt))
    capped = AP.capped_held(AP.Site(toks, tgt, 20, alt), 3, 0)
    assert capped.alt is not None and len(capped.alt) == 3 and AP.capped_held(AP.Site(toks, tgt, 20), 3, 0).alt is None


def test_key_inputs_default_groups_at_half_and_key_at_groups_there(monkeypatch: pytest.MonkeyPatch) -> None:
    from commute import perturb as PT

    toks = torch.tensor([[1, 7], [1, 8], [2, 7]])
    x = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 1.0]])
    monkeypatch.setattr(PT, "window_inputs", lambda *_: x)
    assert torch.equal(PT.key_inputs(None, toks, 1, [7, 8]), torch.tensor([[3.0, 3.0], [0.5, 0.0]]))
    assert torch.equal(PT.key_inputs(None, toks, 1, [1, 2], 0), torch.tensor([[2.0, 5.0], [0.0, 1.0]]))


def test_trigger_table_keys_on_the_previous_token_and_filters_by_the_mode() -> None:
    """Echo's top-1 at 20 is the token at 21. Previous token 5 answers 3 on 3/4 of its windows; 6 always answers 4; 7 has too few
    windows; 8 answers a token outside the values."""
    ID = load_experiment("new_vs_reuse", "input_dep", "apparatus", "reading", "rules")
    t, rows = 2, torch.zeros((300, 512), dtype=torch.long)
    for r in range(300):
        prev, nxt = (5, 3 if r % 4 else 1) if r < 100 else (6, 4) if r < 200 else (7, 3) if r < 220 else (8, 1)
        rows[r, 30:33] = torch.tensor([prev, t, nxt])
    tb = ID.trigger_table(Echo(9), rows, t, [3, 4], 0)
    assert {k: a for k, a in zip(tb.keys, tb.answers, strict=True)} == {5: 3, 6: 4}
    by = {p["key"]: p for p in tb.per_key}
    assert 7 not in by and not by[8]["kept"] and by[5]["kept_share"] == pytest.approx(0.75, abs=0.02)
    assert bool((tb.train[:, 19] == torch.tensor(tb.keys)[(tb.train_answer == 4).long()]).all())
    assert bool((tb.train[:, 21] == tb.train_answer).all()) and bool((tb.held[:, 21] == tb.held_answer).all())
    assert all(p["n_train"] <= 80 and p["n_held"] <= 20 for p in tb.per_key)


def test_common_values_always_holds_the_target() -> None:
    ID = load_experiment("new_vs_reuse", "input_dep", "apparatus", "reading", "rules")
    assert ID.common_values(torch.tensor([1] * 5 + [2] * 3), 9).tolist() == [1, 2, 9]
    assert ID.common_values(torch.tensor([v for v in range(10) for _ in range(20 - v)]), 9).tolist() == [0, 1, 2, 3, 4, 5, 6, 9]


def test_every_saved_record_points_at_a_file() -> None:
    AP = load_experiment("new_vs_reuse", "apparatus")
    rec = AP.records()
    assert len(rec) == 2 + 4 + 2 + 6 + 2 + 6 + 1 + 6
    assert all(p.exists() for p in rec), [p for p in rec if not p.exists()]
