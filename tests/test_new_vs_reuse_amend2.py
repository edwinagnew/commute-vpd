"""E90 Amendment 2's filed rules (experiments/new_vs_reuse/rules_amend2.py, results/predictions_amend2.json) on hand-made numbers, on
stage 2's saved numbers, and amend2.py's own path on a stand-in model. No model is loaded."""

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

from commute import readout as RO
from commute import removal as RM
from commute.jobs import load_experiment

A = load_experiment("new_vs_reuse", "rules_amend2")
R = A.rules
RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "new_vs_reuse" / "results"
KEY = [90, 1, 1]
nan = math.nan


def residuals(spectrum: list[float], n: int = 4000, values: int = 9, d: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows with the given variances on random orthonormal axes, and a value index cycling over `values`."""
    gen = torch.Generator().manual_seed(0)
    axes = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))[0][:, : len(spectrum)]
    z = torch.randn(n, len(spectrum), generator=gen, dtype=torch.float64) * torch.tensor(spectrum, dtype=torch.float64).sqrt()
    return (z @ axes.T).float(), torch.arange(n) % values


def power_law(m: int = 64) -> list[float]:
    return [1.0 / (i + 1) for i in range(m)]


def pcs(h: torch.Tensor, r: int) -> torch.Tensor:
    return A.spectrum(h)[1][:, :r].float()


# ---- the draws ----


def test_fables_smallest_k_is_always_r_so_the_filed_rule_takes_the_largest() -> None:
    """The departure from the brief: the mean share never rises with k, so "the smallest k at or over 0.9 of the basis's" is r on any
    spectrum and for any basis, and every draw would span one subspace."""
    for spec in (power_law(), [1.0] * 64, [10.0, 1.0, 0.1] * 20):
        lam = A.spectrum(residuals(spec)[0])[0]
        for r in (1, 3, 15):
            means = [A.expected_share(lam, r, k) for k in range(r, 65)]
            assert all(a >= b - 1e-12 for a, b in itertools.pairwise(means))
            share = float(lam[:r].sum() / lam.sum())  # the variance-optimal basis; any other holds less
            assert next(k for k in range(r, 65) if A.expected_share(lam, r, k) >= 0.9 * share) == r
            assert A.largest_k(lam, r, 0.9 * share, 20, KEY) > r or A.expected_share(lam, r, r + 1) < 0.9 * share


def test_the_draws_share_is_the_realised_share_on_the_residuals_and_meets_the_bar() -> None:
    h, y = residuals(power_law())
    means = 0.3 * torch.randn(9, 64, generator=torch.Generator().manual_seed(2))
    h = h + means[y]
    u = RM.centred_basis(means.T)  # a value form with real signal, so its split-half bar is high
    rec, draws = A.matched(h, y, u, 20, KEY)
    x = h - h.mean(0)
    direct = [float((x @ d).pow(2).sum() / x.pow(2).sum()) for d in draws]
    assert direct == pytest.approx(rec["draw_shares"], abs=1e-4)
    assert sum(rec["draw_shares"]) / 20 >= rec["target"] == pytest.approx(0.9 * rec["basis_share"])
    assert rec["pairwise_overlap"] == pytest.approx(rec["chance_overlap"], abs=0.1)
    assert all(d.shape == (64, 8) and d.dtype == torch.float32 for d in draws)
    assert rec["k"] > 8 and rec["reason"] is None and rec["bar"] > rec["pairwise_overlap"]
    assert rec["top_r_share"] >= rec["basis_share"] - 1e-6, "Ky Fan"


def test_a_value_form_with_no_signal_has_a_low_bar_and_no_band() -> None:
    """Values unrelated to the residual: the basis is noise, its two halves do not agree, and no draw set can be told from it."""
    h, y = residuals(power_law())
    assert A.split_half(h, y) < 0.5


def test_k_equal_r_every_draw_spans_one_subspace_and_is_no_band() -> None:
    """A basis on the top-r subspace of a residual with a steep fall after r: only k = r reaches 0.9 of its share."""
    h, y = residuals([100.0] * 5 + [0.01] * 40)
    rec, draws = A.matched(h, y, pcs(h, 5), 20, KEY)
    assert rec["k"] == 5 and "k = r" in rec["reason"]
    assert rec["pairwise_overlap"] == pytest.approx(1.0) and len(draws) == 20
    assert A.read_block(state_rows([0.5], [0.0], 0.1), ["x_s0"], 1, True, rec["reason"])["verdicts"]["x"]["verdict"] == A.NO_MATCH


def test_near_degenerate_draws_are_decided_by_the_split_half_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    h, y = residuals([100.0] * 15 + [90.0] + [0.01] * 30)
    monkeypatch.setattr(A, "split_half", lambda *_a: 0.5)
    rec, draws = A.matched(h, y, pcs(h, 15), 20, KEY)
    assert rec["k"] > 15 and rec["pairwise_overlap"] == pytest.approx(15 / rec["k"], abs=0.03)
    assert "bar 0.5" in rec["reason"] and len(draws) == 20, "the draws are still read, beside"
    monkeypatch.setattr(A, "split_half", lambda *_a: 0.999)
    assert A.matched(h, y, pcs(h, 15), 20, KEY)[0]["reason"] is None
    monkeypatch.setattr(A, "split_half", lambda *_a: rec["pairwise_overlap"])
    assert A.matched(h, y, pcs(h, 15), 20, KEY)[0]["reason"] is not None, "a tie is not under the bar"


def test_split_half_bar_edges() -> None:
    h, y = residuals(power_law())
    assert 0 < A.split_half(h, y) < 1
    assert math.isnan(A.split_half(h, torch.zeros(len(h), dtype=torch.long)))
    y2 = y.clone()
    y2[y2 == 3] = 2
    y2[0] = 3  # one window of value 3 can't be split
    assert math.isnan(A.split_half(h, y2))


def test_a_nan_bar_is_no_matched_band_not_a_pass() -> None:
    h = residuals(power_law())[0]
    rec, draws = A.matched(h, torch.zeros(len(h), dtype=torch.long), pcs(h, 3), 20, KEY)
    assert math.isnan(rec["bar"]) and rec["reason"] is not None and len(draws) == 20


def test_rank_above_the_residuals_rank_gives_no_draws() -> None:
    h, y = residuals([1.0, 1.0, 1.0])
    rec, draws = A.matched(h, y, RM.random_basis(64, 5, [2]), 20, KEY)
    assert draws == [] and rec["residual_rank"] == 3 and "exceeds" in rec["reason"] and rec["k"] is None


def test_zero_variance_is_the_zero_denominator() -> None:
    h = torch.ones(100, 64)
    rec, draws = A.matched(h, torch.arange(100) % 4, RM.random_basis(64, 3, [2]), 20, KEY)
    assert draws == [] and "no variance" in rec["reason"]


def test_a_basis_with_no_variance_is_matched_by_anything_and_takes_every_component() -> None:
    h, y = residuals([1.0] * 10)
    empty = A.spectrum(h)[1][:, -2:].float()  # the two smallest components: no variance in them
    rec, _ = A.matched(h, y, empty, 20, KEY)
    assert rec["basis_share"] == pytest.approx(0, abs=1e-6) and rec["k"] == 10


def test_stage1s_random_draws_read_as_k_equal_d_and_fail_the_match() -> None:
    """The old unmatched band's draws: r/768 of the variance, as the formula says, and far under 0.9 of the basis's."""
    c = json.loads((RESULTS / "stage1.json").read_text())["control"]
    for name, r in (("W0's guess", 15), ("token identity", 31)):
        for b in ("1", "2"):
            assert c[name][b]["variance_share_random"] == pytest.approx(r / 768, rel=0.05)
            assert c[name][b]["variance_share_random"] < A.MATCH_RATIO * c[name][b]["variance_share"]


# ---- the verdict ----


def state_rows(shares: list[float], writes: list[float], band: float) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for i, (s, w) in enumerate(zip(shares, writes, strict=True)):
        rows[f"x_s{i}"] = {"1": {"A_lo": s, "band": band}}
        rows[f"x_s{i}_write"] = {"1": {"A_lo": w, "band": band}}
    return rows


def test_read_block_is_rules_verdict_when_a_band_exists() -> None:
    r = A.read_block(state_rows([0.5, 0.6], [0.0, 0.0], 0.1), ["x_s0", "x_s1"], 1, True, None)
    assert r["verdicts"]["x"]["verdict"] == R.REUSES and r["status"]["x"] == A.ABOVE and r["band"] == 0.1
    r = A.read_block(state_rows([0.5, 0.05], [0.0, 0.0], 0.1), ["x_s0", "x_s1"], 1, True, None)
    assert r["verdicts"]["x"]["verdict"] == "mixed" and r["status"]["x"] == A.MIXED
    assert A.read_block(state_rows([0.1], [0.0], 0.1), ["x_s0"], 1, True, None)["status"]["x"] == A.WITHIN, "a tie is within"


def test_no_teeth_is_unreadable_and_the_rows_are_printed_beside() -> None:
    r = A.read_block(state_rows([0.5], [0.0], 0.1), ["x_s0"], 1, False, None)
    assert r["verdicts"]["x"]["verdict"] == R.NO_TEETH and r["beside"]["x"]["verdict"] == R.REUSES and r["status"]["x"] == A.UNREADABLE


def test_no_matched_band_wins_over_teeth_and_reads_beside_when_there_are_draws() -> None:
    r = A.read_block(state_rows([0.5], [0.0], 0.1), ["x_s0"], 1, False, "k = r")
    assert r["verdicts"]["x"]["verdict"] == A.NO_MATCH and r["beside"]["x"]["verdict"] == R.REUSES and r["status"]["x"] == A.UNREADABLE


def test_a_block_with_no_draws_reads_no_match_and_no_gain_beside() -> None:
    states = {"x_s0": {"G_logodds": 3.0}, "x_s0_write": {"G_logodds": 3.0}}
    r = A.read_block(states, ["x_s0"], 1, False, "rank 5 exceeds")
    assert r["verdicts"]["x"]["verdict"] == A.NO_MATCH and math.isnan(r["band"]) and r["status"]["x"] == A.UNREADABLE


def test_do_nothing_cannot_pass_or_trigger_a_branch() -> None:
    """Gain 0: A_lo and every draw nan under the floor. It reads "no gain", not "within": a zero update must not satisfy (i)."""
    r = A.read_block({"x_s0": {"1": {"A_lo": nan, "band": nan}}, "x_s0_write": {"1": {"A_lo": nan, "band": nan}}}, ["x_s0"], 1, True, None)
    assert r["verdicts"]["x"]["verdict"] == R.NO_GAIN and r["status"]["x"] == A.UNREADABLE
    assert R.verdict(True, nan, 0.0, 0.0) == R.NO_GAIN
    assert A.branch(dict.fromkeys(A.VALUE_FORM, A.UNREADABLE))["branch"] == A.NO_READING


def test_a_gain_about_zero_cancels_in_the_comparison_above_the_floor() -> None:
    """Above the floor, state and draws divide by one G, so a small gain does not move a single state's verdict; under it, no gain."""
    w0 = torch.zeros(3)
    read = lambda g: RO.shares((w0, w0), (w0, w0), (w0 + g, w0 + g), (w0 + g - 0.5, w0 + g - 0.5), 0.05)["A_lo"]
    assert read(0.06) == pytest.approx(0.5 / 0.06, rel=1e-4) and math.isnan(read(0.04)) and math.isnan(read(0.0))


def test_the_old_bands_numbers_reproduce_stage2s_verdicts_and_every_row_is_above_them() -> None:
    """Stage 2's saved numbers through read_block: the old unmatched band passes the five rows, which is the question, and its draws
    fail the match (test_stage1s_random_draws...), so no row can reach "reuses C" under this rule on those draws."""
    s2 = json.loads((RESULTS / "stage2.json").read_text())
    forms = {"guess": ["rank1", "cipher", "elicit"], "identity": ["ident32", "rand32"], "copy24": ["copy24"]}
    seeds = {"rank1": 3, "cipher": 3, "elicit": 3, "ident32": 2, "rand32": 2, "copy24": 3}
    status: dict[str, str] = {}
    for form, maps in forms.items():
        names = [f"{m}_s{i}" for m in maps for i in range(seeds[m])]
        r = A.read_block(s2["states"], names, 1, True, None)
        assert r["band"] == pytest.approx(s2["rows"][form]["band"])
        assert {m: v["verdict"] for m, v in r["verdicts"].items()} == {m: v["verdict"] for m, v in s2["rows"][form]["verdicts"].items()}
        status |= r["status"]
    assert all(status[m] == A.ABOVE for m in A.VALUE_FORM)
    assert A.branch(status)["branch"] == A.CONTROL_FAILS, "as the old numbers stand: rand32 is above, Fable's question 2"


def test_the_old_numbers_at_block_2_reproduce_the_checks_table() -> None:
    """Finding 4's re-derivation of block 2, which `stage_2_band` prints beside each verdict."""
    s = json.loads((RESULTS / "stage2.json").read_text())["states"]
    read = lambda maps, n: A.read_block(s, [f"{m}_s{i}" for m in maps for i in range(n)], 2, True, None)
    guess, identity, copy24 = read(["rank1", "cipher", "elicit"], 3), read(["ident32", "rand32"], 2), read(["copy24"], 3)
    assert guess["band"] == pytest.approx(0.0299, abs=1e-4) and identity["band"] == pytest.approx(0.0613, abs=1e-4) and copy24["band"] == pytest.approx(0.0385, abs=1e-4)
    got = {m: v["verdict"] for r in (guess, identity, copy24) for m, v in r["verdicts"].items()}
    assert got == {"rank1": R.NOT_SPECIFIC, "cipher": R.NO_REACH, "elicit": R.NO_REACH, "ident32": R.NO_REACH, "rand32": R.REUSES, "copy24": "mixed"}


def test_branches_cover_every_case() -> None:
    w, a, m, u = A.WITHIN, A.ABOVE, A.MIXED, A.UNREADABLE
    st = lambda **k: {**dict.fromkeys(A.VALUE_FORM, w), **k}
    assert A.branch(st())["branch"] == A.ALL_WITHIN
    assert A.branch(st(ident32=a, copy24=a))["branch"] == A.COPY_SPECIFIC
    assert A.branch(st(ident32=a, copy24=a, rank1=a))["branch"] == A.COPY_SPECIFIC, "(ii) reads the identity rows; guess rows read by their own verdicts"
    b = A.branch(st(rand32=a))
    assert b["branch"] == A.CONTROL_FAILS and set(b["beside"]) == {"ident32", "copy24"}
    assert A.branch(st(ident32=a))["branch"] == A.OTHER
    assert A.branch(st(rand32=m))["branch"] == A.OTHER
    assert A.branch(st(rank1=a))["branch"] == A.OTHER
    assert A.branch(st(rand32=u, ident32=u))["branch"] == A.OTHER, "a part unreadable is not (i)"
    assert A.branch(st(rand32=u, ident32=a, copy24=a))["branch"] == A.OTHER
    assert A.branch(st(ident32=u, copy24=u, rand32=a))["branch"] == A.CONTROL_FAILS


def test_task_match_edges() -> None:
    assert A.task_match(0.2, 0.2) == "matched"
    assert A.task_match(0.2, 0.19) == "matched" and A.task_match(0.2, 0.21) == "matched"
    assert A.task_match(0.2, 0.1) == "under"
    assert A.task_match(0.2, 0.3) == "over"
    assert A.task_match(0.0, 0.1) == "no share" and A.task_match(nan, 0.1) == "no share" and A.task_match(0.2, nan) == "no share"


def test_each_verdict_names_its_block() -> None:
    assert A.named({2: R.NO_REACH, 1: R.REUSES}) == f"{R.REUSES} at block 1; {R.NO_REACH} at block 2"


def test_the_filing_states_what_the_code_uses() -> None:
    P = json.loads((RESULTS / "predictions_amend2.json").read_text())
    assert P["rule"]["match_ratio"] == A.MATCH_RATIO and P["outcome_no_band"] == A.NO_MATCH
    A2 = load_experiment("new_vs_reuse", "amend2")
    assert P["reproduction"]["tolerance"] == A2.REPRODUCE_TOL
    assert set(P["branches"]) >= {A.ALL_WITHIN, A.COPY_SPECIFIC, A.CONTROL_FAILS, A.NO_READING, A.OTHER}


# ---- amend2.py's own path on a stand-in model ----


class Noisy:
    """A stand-in W0: the residual is the token's embedding plus noise from the tokens before it; a "state" is a number that scales the logits."""

    device = "cpu"

    def __init__(self, vocab: int, d: int) -> None:
        gen = torch.Generator().manual_seed(0)
        self.emb, self.W = torch.randn(vocab, d, generator=gen), torch.randn(d, vocab, generator=gen)
        self.noise, self.scale = torch.randn(64, d, generator=gen), 0.0

    @contextmanager
    def at(self, dW: float | None) -> Iterator[None]:
        self.scale = dW or 0.0
        yield
        self.scale = 0.0

    def run(self, toks: torch.Tensor, edit: dict | None = None, capture: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.emb[toks] + self.noise[(toks.roll(1, 1) * 7 + toks.roll(2, 1)) % 64]
        c = {}
        for name in RO.RESID:
            c[name] = h
            if edit and name in edit:
                h = edit[name](h)
        return (h @ self.W) * (1 + self.scale), c


def test_the_job_path_reads_controls_rows_and_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    A2 = load_experiment("new_vs_reuse", "amend2")
    AP, RD, S2 = A2.AP, A2.RD, A2.S2
    net: Any = Noisy(12, 64)
    S = {"n_random": 20, "random_seed": 90, "gain_floor": 0.05, "blocks": [1, 2], "control_seed": 1}
    j = RD.Job(net, S, {}, {"sha256": {}}, tmp_path / "out.json")
    monkeypatch.setattr(A2, "OUT", tmp_path / "out.json")
    toks = torch.randint(0, 9, (800, 41), generator=torch.Generator().manual_seed(1))
    y = toks[:, 20]
    h = RO.cache64(net, toks, torch.tensor([1]), 20)[0]
    c = AP.Capability("token identity", h, y, AP.own_top1(net, AP.Site(toks[:200], y[:200], 20)), 8)
    bases = {b: u.float() for b, u in c.bases([1, 2]).items()}
    f = A2.matched_basis(j, "identity", c, bases, 2)
    assert set(f.bases) == {1, 2} and len(f.randoms[1]) == 20 and j.res["matched"]["identity"]["1"]["k"] >= 8

    w0 = AP.read_all(net, c.read, None, f.bases, f.randoms)
    ref = {str(b): {"drop": float((w0["clean"][1] - w0[b, -1][1]).mean()), "variance_share": AP.variance_share(c.fit_h[b], bases[b]), "teeth": True} for b in (1, 2)}
    on = A2.controls(j, "identity", c, f, ref)
    assert set(on) == {"1", "2"} and j.res["do_nothing"]["identity_1"]["verdict"] in (R.NO_GAIN, R.NO_TEETH)
    assert math.isnan(j.res["do_nothing"]["identity_1"]["A_lo"])

    site = AP.Site(toks[200:300], y[200:300], 20)
    names = [f"{m}_s0" for m in A.VALUE_FORM]
    monkeypatch.setattr(S2, "rows_of", lambda: {"identity": [(n, tmp_path / "windows_x.pt", tmp_path / f"{n}.pt") for n in names]})
    monkeypatch.setattr(j, "site", lambda _p: (site, None))
    monkeypatch.setattr(j, "state", lambda _p: 1.0)
    monkeypatch.setattr(A2, "saved_write", lambda *_a: 0.5)
    j2 = RD.Job(net, S, {}, {"sha256": {}}, tmp_path / "second.json")
    w0s = AP.read_all(net, site, None, f.bases, f.randoms)
    for n in names:
        RD.read_state(j2, n, site, 1.0, w0s, f)
        RD.read_state(j2, f"{n}_write", site, 0.5, w0s, f)
    second = {"states": j2.res["states"]}

    A2.read_form(j, "identity", f, second)  # reproduces, so no stop
    task = j.res["task_windows"]["windows_x"]["1"]
    assert 0 < task["basis"] < 1 and len(task["draws"]) == 20 and task["match"] in ("matched", "under", "over")

    A2.table(j, second, {"identity": ref}, {"identity": on})
    row = j.res["table"]["cipher"]
    assert "at block 1" in row["verdict"] and "at block 2" in row["verdict"] and "at block 1" in row["stage_2_band"]
    assert set(j.res["branch"]) == {"1", "2"} and j.res["branch"]["1"]["branch"] in (A.ALL_WITHIN, A.COPY_SPECIFIC, A.CONTROL_FAILS, A.NO_READING, A.OTHER)
    json.dumps(j.res)

    def raised(_o: object, _r: object, why: str) -> None:
        raise RuntimeError(why)

    monkeypatch.setattr(A2, "stop", raised)
    second["states"]["rank1_s0"]["1"]["A_lo"] += 0.01
    with pytest.raises(RuntimeError, match="rank1_s0 block 1"):
        A2.read_form(j, "identity", f, second)
