"""E95's filed rules (experiments/token_identity/read_rules.py, results/predictions.json) through their edge cases, the do-nothing state
and E90 Amendment 2's saved numbers, and run.py's reads on a stand-in model. No model is loaded."""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from commute.jobs import load_experiment

RR = load_experiment("token_identity", "read_rules")
RUN = load_experiment("token_identity", "run")
ROOT = Path(__file__).resolve().parents[1] / "experiments"
SAVED = json.loads((ROOT / "new_vs_reuse/results/amend2.json").read_text())["states"]
FILED = json.loads((ROOT / "token_identity/results/predictions.json").read_text())
nan = math.nan
NULL = {"lo": -0.1, "hi": 0.1}


def row(status: str, bites: bool = True) -> dict[str, Any]:
    return {"status": status, "write_bites": bites}


def number(x: Any) -> float:
    return nan if x == "nan" else float(x)


def saved_row(name: str) -> dict[str, Any]:
    r = SAVED[name]["1"]
    return {"A_lo": number(r["A_lo"]), "random": [number(x) for x in r["random"]], "band": number(r["band"])}


# ---- small pieces ----


def test_percentile_is_signed_and_skips_nan() -> None:
    assert RR.percentile([-3.0, -2.0, -1.0], 95) < 0, "unlike band_of, whose absolute value would read 2.9"
    assert RR.percentile([1.0, nan], 95) == 1.0
    assert math.isnan(RR.percentile([], 95)) and math.isnan(RR.percentile([nan], 95))


def test_over_seeds_edges() -> None:
    assert RR.over_seeds([]) == RR.UNREADABLE
    assert RR.over_seeds([RR.ABOVE]) == RR.ABOVE, "a one-element pool reads as itself"
    assert RR.over_seeds([RR.ABOVE, RR.WITHIN]) == RR.MIXED
    assert RR.over_seeds([RR.ABOVE, RR.UNREADABLE]) == RR.UNREADABLE


def test_write_works_needs_a_match_and_a_gain_over_the_floor() -> None:
    assert RR.write_works("ok", 13.0, [-2.0, 1.0])["works"]
    assert not RR.write_works("ok", 1.0, [1.0, 1.0])["works"], "a tie is not over"
    assert not RR.write_works("ok", 13.0, [])["works"], "no floor"
    assert not RR.write_works("ok", nan, [1.0])["works"]
    assert not RR.write_works("unreachable", 13.0, [1.0])["works"]
    assert not RR.write_works("ok", 13.0, [14.0, 14.0])["works"], "a write that raises every answer has not learned its table"


# ---- read 1 ----


def test_seed_row_ties_nans_and_a_write_that_failed() -> None:
    assert RR.seed_row(0.5, 0.4, [0.0], NULL, True)["status"] == RR.EQUAL, "a gap equal to half the range is about equal"
    assert RR.seed_row(0.5, 0.2, [0.0], NULL, True)["status"] == RR.ABOVE
    assert RR.seed_row(0.2, 0.5, [0.0], NULL, True)["status"] == RR.BELOW
    wide = {"lo": 0.2, "hi": 0.4}
    assert RR.seed_row(0.5, 0.2, [0.0], wide, True)["status"] == RR.INSIDE, "0.3 is more than half the range, and what the draws give"
    assert RR.seed_row(0.5, 0.1, [0.0], wide, True)["status"] == RR.INSIDE, "a gap equal to the top limit does not cross"
    assert RR.seed_row(0.5, 0.05, [0.0], wide, True)["status"] == RR.ABOVE
    assert RR.seed_row(0.5, 0.45, [0.0], wide, True)["status"] == RR.EQUAL, "0.05 is under half the range, which comes first"
    assert RR.seed_row(0.5, 0.35, [0.0], wide, True)["status"] == RR.BELOW
    assert RR.seed_row(0.5, 0.2, [0.0], NULL, False)["status"] == RR.UNREADABLE
    assert RR.seed_row(0.5, nan, [nan], NULL, True)["status"] == RR.UNREADABLE
    assert RR.seed_row(0.5, 0.2, [0.0], {"lo": nan, "hi": nan}, True)["status"] == RR.UNREADABLE
    assert not RR.seed_row(0.5, nan, [nan], NULL, True)["write_bites"]


@pytest.mark.parametrize("copy,rand,bites,want", [
    (RR.EQUAL, RR.EQUAL, True, RR.TOKEN_ONLY), (RR.BELOW, RR.EQUAL, True, RR.TOKEN_ONLY), (RR.EQUAL, RR.BELOW, True, RR.TOKEN_ONLY), (RR.BELOW, RR.BELOW, True, RR.TOKEN_ONLY),
    (RR.EQUAL, RR.EQUAL, False, RR.NO_CALL_SPARED), (RR.BELOW, RR.BELOW, False, RR.NO_CALL_SPARED),
    (RR.ABOVE, RR.EQUAL, True, RR.COPY_MORE), (RR.ABOVE, RR.BELOW, False, RR.COPY_MORE), (RR.ABOVE, RR.INSIDE, False, RR.COPY_MORE),
    (RR.ABOVE, RR.ABOVE, True, RR.BOTH_MORE), (RR.ABOVE, RR.ABOVE, False, RR.BOTH_MORE),
    (RR.EQUAL, RR.ABOVE, True, RR.RANDOM_MORE), (RR.BELOW, RR.ABOVE, False, RR.RANDOM_MORE), (RR.INSIDE, RR.ABOVE, True, RR.RANDOM_MORE),
    (RR.INSIDE, RR.EQUAL, True, RR.NO_CALL_SIZE), (RR.EQUAL, RR.INSIDE, True, RR.NO_CALL_SIZE), (RR.INSIDE, RR.INSIDE, False, RR.NO_CALL_SIZE),
])  # fmt: skip
def test_read1_outcomes(copy: str, rand: str, bites: bool, want: str) -> None:
    got = RR.read1(True, [row(copy, bites), row(copy, bites)], [row(rand, bites), row(rand, bites)], RR.ALL_PAIRS)
    assert got["outcome"] == want


def test_copy_above_needs_its_difference_from_random_to_be_beyond_the_draws() -> None:
    """Copy just over its limit and random just under it would otherwise read as a separation."""
    for call in (RR.SOME_PAIRS, RR.NO_PAIR, RR.UNREADABLE):
        assert RR.read1(True, [row(RR.ABOVE)], [row(RR.EQUAL)], call)["outcome"] == RR.NO_CALL_DIFF
    assert RR.read1(True, [row(RR.ABOVE)], [row(RR.ABOVE)], RR.NO_PAIR)["outcome"] == RR.BOTH_MORE, "no separation is claimed there"


def test_equal_and_below_on_two_seeds_are_not_a_disagreement() -> None:
    got = RR.read1(True, [row(RR.EQUAL), row(RR.BELOW)], [row(RR.EQUAL), row(RR.EQUAL)], RR.NO_PAIR)
    assert got["outcome"] == RR.TOKEN_ONLY and got["copy"] == RR.AT_MOST


def test_read1_lands_every_case_in_one_outcome() -> None:
    outcomes = {RR.TOKEN_ONLY, RR.COPY_MORE, RR.BOTH_MORE, RR.RANDOM_MORE, RR.NO_CALL_WRITE, RR.NO_CALL_SEEDS, RR.NO_CALL_SPARED, RR.NO_CALL_SIZE, RR.NO_CALL_DIFF}
    seen = set()
    statuses = (RR.ABOVE, RR.EQUAL, RR.BELOW, RR.INSIDE, RR.UNREADABLE)
    for a, b, c, d, bites, teeth, call in itertools.product(statuses, statuses, statuses, statuses, (True, False), (True, False), (RR.ALL_PAIRS, RR.SOME_PAIRS, RR.NO_PAIR, RR.UNREADABLE)):
        seen.add(RR.read1(teeth, [row(a, bites), row(b, bites)], [row(c, bites), row(d, bites)], call)["outcome"])
    assert seen == outcomes
    assert outcomes == set(FILED["read1"]["outcomes"]), "the filing names every outcome"


def test_read1_no_call_cases() -> None:
    assert RR.read1(False, [row(RR.ABOVE)], [row(RR.EQUAL)], RR.ALL_PAIRS)["outcome"] == RR.NO_CALL_WRITE, "no teeth"
    assert RR.read1(True, [], [], RR.UNREADABLE)["outcome"] == RR.NO_CALL_WRITE, "no seeds"
    assert RR.read1(True, [row(RR.UNREADABLE), row(RR.ABOVE)], [row(RR.EQUAL)], RR.ALL_PAIRS)["outcome"] == RR.NO_CALL_WRITE
    assert RR.read1(True, [row(RR.ABOVE), row(RR.EQUAL)], [row(RR.EQUAL)], RR.ALL_PAIRS)["outcome"] == RR.NO_CALL_SEEDS


def test_do_nothing_cannot_pass_either_read() -> None:
    """Gain 0: `readout.shares` gives nan for the share and every draw."""
    draws = [nan] * 20
    null = RR.gaps_under_draws(draws, draws)
    r = RR.seed_row(nan, nan, draws, null, RR.write_works("no_target", nan, [])["works"])
    assert r["status"] == RR.UNREADABLE and not r["write_bites"]
    assert RR.read1(True, [r, r], [r, r], RR.UNREADABLE)["outcome"] == RR.NO_CALL_WRITE
    states = {"ident32b_s0": {"A_lo": nan, "random": draws}, "rand32b_s0": {"A_lo": nan, "random": draws}}
    pair = RR.paired({"s0": states["ident32b_s0"]}, {"s0": states["rand32b_s0"]})
    assert pair["call"] == RR.UNREADABLE
    assert RR.own_bands([nan, nan], [nan, nan], True) == RR.UNREADABLE
    assert RR.read2(RR.UNREADABLE, RR.UNREADABLE, pair["call"]) == RR.NO_CALL_REPLICATE


def test_a_write_identical_to_the_state_reads_about_equal() -> None:
    """The negative control of the rule: the state against itself, on Amendment 2's saved draws."""
    s = saved_row("ident32_s0")
    r = RR.seed_row(s["A_lo"], s["A_lo"], s["random"], RR.gaps_under_draws(s["random"], s["random"]), True)
    assert r["status"] == RR.EQUAL and r["gap"] == 0


def test_a_spared_write_reads_above_on_saved_numbers() -> None:
    """The positive control of the rule, no GPU: block 1's write is upstream of the removal and spared, so the rule must say "above" on
    copy against it. On random it stays inside the draws although the gap is 0.36: the limit of "not above", filed in `blind_to`."""
    rows, gaps = {}, {}
    for t in ("ident32", "rand32"):
        rows[t], gaps[t] = [], {}
        for seed in (0, 1):
            s, w = saved_row(f"{t}_s{seed}"), saved_row(f"{t}_s{seed}_write")
            null = RR.gaps_under_draws(s["random"], w["random"])
            rows[t].append(RR.seed_row(s["A_lo"], w["A_lo"], w["random"], null, True))
            gaps[t][f"s{seed}"] = {"A_lo": rows[t][-1]["gap"], "random": null["draw_gaps"]}
    got = RR.read1(True, rows["ident32"], rows["rand32"], RR.paired(gaps["ident32"], gaps["rand32"])["call"])
    assert (got["copy"], got["random"], got["outcome"]) == (RR.ABOVE, RR.INSIDE, RR.COPY_MORE), (got, rows)
    assert not got["write_bites"], "block 1's write is spared"
    assert [round(r["gap"], 3) for r in rows["rand32"]] == [0.364, 0.352] and [round(r["hi"], 3) for r in rows["rand32"]] == [0.385, 0.389], "as filed"
    assert [round(r["gap"], 3) for r in rows["ident32"]] == [0.542, 0.573] and [round(r["hi"], 3) for r in rows["ident32"]] == [0.481, 0.494], "as filed"


# ---- read 2 ----


def test_paired_reproduces_the_checks_table() -> None:
    """Finding 2 of reports/reviews/2026-11-05_review_e90-amend2-check.md, block 1, on table A; table A is a test here and counts for nothing."""
    copy = {f"s{s}": saved_row(f"ident32_s{s}") for s in (0, 1)}
    rand = {f"s{s}": saved_row(f"rand32_s{s}") for s in (0, 1)}
    got = RR.paired(copy, rand)
    want = {"s0 - s0": (0.129, 0.035, 0.099), "s1 - s1": (0.171, 0.045, 0.112), "s0 - s1": (0.140, 0.027, 0.097), "s1 - s0": (0.160, 0.053, 0.118)}
    for k, (gap, mean, top) in want.items():
        p = got["pairs"][k]
        assert (round(p["gap"], 3), round(p["draws_mean"], 3), round(p["draws_max"], 3)) == (gap, mean, top)
    assert got["call"] == RR.ALL_PAIRS


def test_paired_calls() -> None:
    x = {"A_lo": 0.5, "random": [0.1, 0.2]}
    assert RR.paired({"a": x}, {"b": {"A_lo": 0.1, "random": [0.1, 0.1]}})["call"] == RR.ALL_PAIRS
    assert RR.paired({"a": x}, {"b": {"A_lo": 0.4, "random": [0.0, 0.0]}})["call"] == RR.NO_PAIR, "0.1 against a largest draw gap of 0.2"
    assert RR.paired({"a": {"A_lo": 0.3, "random": [0.2]}}, {"b": {"A_lo": 0.1, "random": [0.0]}})["call"] == RR.NO_PAIR, "a tie is not above"
    assert RR.paired({"a": x, "c": {"A_lo": 0.1, "random": [0.1, 0.2]}}, {"b": {"A_lo": 0.1, "random": [0.1, 0.1]}})["call"] == RR.SOME_PAIRS
    assert RR.paired({}, {})["call"] == RR.UNREADABLE
    assert RR.paired({"a": x}, {"b": {"A_lo": 0.1, "random": [nan, nan]}})["call"] == RR.UNREADABLE


def test_own_bands_on_saved_numbers_read_as_the_check_found() -> None:
    """Finding 1: seed by seed the random map sits over its own band, where the filed largest-over-states band reads it within."""
    for t in ("ident32", "rand32"):
        r = [saved_row(f"{t}_s{s}") for s in (0, 1)]
        assert RR.own_bands([x["A_lo"] for x in r], [x["band"] for x in r], True) == RR.ABOVE
    assert RR.own_bands([0.5, 0.1], [0.3, 0.3], True) == RR.MIXED
    assert RR.own_bands([0.5, 0.5], [0.3, 0.3], False) == RR.UNREADABLE
    assert RR.own_bands([], [], True) == RR.UNREADABLE


@pytest.mark.parametrize("copy,rand,pair,want", [
    (RR.ABOVE, RR.WITHIN, RR.ALL_PAIRS, RR.CONFIRMED), (RR.ABOVE, RR.WITHIN, RR.SOME_PAIRS, RR.FILED_ONLY), (RR.ABOVE, RR.WITHIN, RR.NO_PAIR, RR.FILED_ONLY),
    (RR.WITHIN, RR.WITHIN, RR.ALL_PAIRS, RR.NOT_REPEATED), (RR.WITHIN, RR.WITHIN, RR.NO_PAIR, RR.NOT_REPEATED),
    (RR.ABOVE, RR.ABOVE, RR.ALL_PAIRS, RR.RANDOM_ABOVE), (RR.WITHIN, RR.ABOVE, RR.NO_PAIR, RR.RANDOM_ABOVE), (RR.MIXED, RR.ABOVE, RR.NO_PAIR, RR.RANDOM_ABOVE),
    (RR.MIXED, RR.WITHIN, RR.ALL_PAIRS, RR.SEEDS_DIFFER), (RR.ABOVE, RR.MIXED, RR.ALL_PAIRS, RR.SEEDS_DIFFER), (RR.WITHIN, RR.MIXED, RR.NO_PAIR, RR.SEEDS_DIFFER),
    (RR.UNREADABLE, RR.WITHIN, RR.ALL_PAIRS, RR.NO_CALL_REPLICATE), (RR.ABOVE, RR.UNREADABLE, RR.ALL_PAIRS, RR.NO_CALL_REPLICATE), (RR.ABOVE, RR.WITHIN, RR.UNREADABLE, RR.NO_CALL_REPLICATE),
])  # fmt: skip
def test_read2_outcomes(copy: str, rand: str, pair: str, want: str) -> None:
    assert RR.read2(copy, rand, pair) == want


def test_read2_every_case_is_filed() -> None:
    statuses = (RR.ABOVE, RR.WITHIN, RR.MIXED, RR.UNREADABLE)
    seen = {RR.read2(a, b, c) for a, b, c in itertools.product(statuses, statuses, (RR.ALL_PAIRS, RR.SOME_PAIRS, RR.NO_PAIR, RR.UNREADABLE))}
    assert seen == set(FILED["read2"]["outcomes"])


# ---- the filing against the code ----


def test_the_filing_names_the_codes_settings() -> None:
    S = FILED["settings"]
    assert S["write_key"] == RUN.WRITE_2 and S["tag_a"] == RUN.TAG_A and S["tag_b"] == RUN.TAG_B and S["n_rotations"] == len(RUN.ROTATIONS)
    assert FILED["reproduction"]["tolerance"] == RUN.REPRODUCE_TOL
    assert RUN.PT.WRITE_KEY == "h.1.mlp.down_proj"
    with RUN.patch.object(RUN.PT, "WRITE_KEY", RUN.WRITE_2):
        assert RUN.AP.PT.WRITE_KEY == RUN.WRITE_2, "apparatus.hand_write reads the same module's name"
    assert RUN.PT.WRITE_KEY == "h.1.mlp.down_proj"


# ---- run.py's reads on a stand-in model ----


class Stand:
    """A stand-in net, four blocks: each block's MLP input and output follow from its input residual; a "state" scales the logits."""

    device = "cpu"

    def __init__(self, vocab: int, d: int) -> None:
        gen = torch.Generator().manual_seed(0)
        self.emb, self.W, self.scale = torch.randn(vocab, d, generator=gen), torch.randn(d, vocab, generator=gen), 0.0
        self.noise = torch.randn(64, d, generator=gen)

    @contextmanager
    def at(self, dW: float | None) -> Iterator[None]:
        self.scale = dW or 0.0
        yield
        self.scale = 0.0

    def run(self, toks: torch.Tensor, edit: dict | None = None, capture: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h, seen = self.emb[toks] + self.noise[(toks.roll(1, 1) * 7 + toks.roll(2, 1)) % 64], {}
        for b in range(4):
            if edit and f"in{b}" in edit:
                h = edit[f"in{b}"](h)
            seen[f"mlp{b}"] = h.tanh()
            h = h + 0.1 * seen[f"mlp{b}"]
        return (h @ self.W) * (1 + self.scale), seen


def stand_job(tmp_path: Path) -> tuple[Any, Any, Any]:
    net: Any = Stand(12, 64)
    S = {"n_random": 20, "random_seed": 90, "gain_floor": 0.05, "blocks": [1], "control_seed": 1}
    j = RUN.RD.Job(net, S, {}, {"sha256": {}, "states": {}}, tmp_path / "out.json")
    toks = torch.randint(0, 9, (300, 41), generator=torch.Generator().manual_seed(1))
    s = RUN.AP.Site(toks, toks[:, 20], 20)
    x = net.emb[toks[:, 20]] + net.noise[(toks[:, 19] * 7 + toks[:, 18]) % 64]
    u = RUN.AP.RM.centred_basis(torch.stack([x[toks[:, 20] == v].mean(0) for v in range(9)], 1)).float()
    return j, s, RUN.RD.Basis({1: u}, {1: [RUN.AP.RM.random_basis(64, u.shape[1], [1, i]) for i in range(20)]})


def test_placement_reads_the_two_sites_and_stops_when_the_write_is_not_downstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    j, s, f = stand_job(tmp_path)
    monkeypatch.setattr(RUN, "OUT", tmp_path / "out.json")
    once = ~(s.toks[:, :20] == s.toks[:, 20:21]).any(1)
    assert once.sum() > 0
    site = RUN.AP.Site(s.toks[once], s.tgt[once], 20)
    RUN.placement(j, site, f.bases[1])
    assert j.res["placement"]["mlp1_change"] == 0 and j.res["placement"]["mlp2_change"] > 0

    def halted(_r: object, why: str) -> None:
        raise RuntimeError(why)

    monkeypatch.setattr(RUN, "halt", halted)
    with pytest.raises(RuntimeError, match="between the two writes"):
        RUN.placement(j, s, f.bases[1])  # these windows hold the key earlier


def test_stage_order_reads_draws_before_the_basis_and_scores_read_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    j, s, f = stand_job(tmp_path)
    monkeypatch.setattr(RUN, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(RUN, "LARGE", tmp_path)
    w0 = RUN.under_draws(j, s, None, f)
    assert (1, -1) not in w0, "stage 1 has not read the basis"
    rows: Any = {}
    for t, gain in (("ident32", 1.0), ("rand32", 0.5)):
        for seed in (0, 1):
            name = f"{t}_s{seed}"
            st, wr = RUN.under_draws(j, s, gain, f), RUN.under_draws(j, s, gain / 2, f)
            rows[t] = rows.get(t, {}) | {name: (gain, st), f"{name}_write": (gain / 2, wr)}
            j.res.setdefault("thresholds", {})[name] = RUN.RR.gaps_under_draws(RUN.draw_shares(j, w0, st, f), RUN.draw_shares(j, w0, wr, f))
            j.res.setdefault("write_check", {})[name] = {"works": True}
    assert not any(k == "states" and v for k, v in j.res.items()), "no share is scored before the basis"
    for t, r in rows.items():
        RUN.basis_stage(j, f, s, dict(w0), r, t)
    RUN.read_one(j, True)
    assert j.res["read1"]["outcome"] in RUN.RR.__dict__.values()
    assert (tmp_path / "reads_ident32.pt").exists()
    for t, b in (("ident32", "ident32b"), ("rand32", "rand32b")):
        for seed in (0, 1):
            j.res["states"][f"{b}_s{seed}"] = j.res["states"][f"{t}_s{seed}"]
            j.res["states"][f"{b}_s{seed}_write"] = j.res["states"][f"{t}_s{seed}_write"]
    RUN.read_two(j, True, None)
    assert j.res["read2"]["outcome"] in set(FILED["read2"]["outcomes"])
    assert set(j.res["read2"]["filed"]["status"]) == {"ident32b", "rand32b"} and set(j.res["read2"]["own_bands"]) == {"ident32b", "rand32b"}
