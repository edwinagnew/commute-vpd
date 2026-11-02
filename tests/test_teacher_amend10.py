"""CPU check of teacher_training Amendment 10's rules (experiments/teacher_training/amend10.py) at their edges, of its arithmetic on
small random matrices, of the score on made-up reads, and of the filing; no model, no saved state. Needs E12's amend1 importable:
PYTHONPATH=experiments/new_vs_existing ~/commute-agents/fleet/cpu-run --mem 8G make test FILES="tests/test_teacher_amend10.py"
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from commute import split

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_training"
NAN = math.nan


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("run")
_load("amend9")
A = _load("amend10")
M, MD, FLOOR = A.CM.gap_min, A.CM.gap_dist_min, A.CM.rerun_floor
CLOSE, FAR, TELL = A.AS_CLOSE, A.FURTHER, A.TELL


def test_angle_label_edges() -> None:
    for bad in ((NAN, 0.8, 0.1, 1.0, M), (0.5, NAN, 0.1, 1.0, M), (0.5, 0.8, NAN, 1.0, M), (0.5, 0.8, 0.1, NAN, M), (0.5, 0.8, 0.1, 1.0, NAN)):
        assert A.angle_label(*bad) == "unscorable"
    assert A.angle_label(0.9, FLOOR - 1e-9, -0.5, 1.0, M) == TELL  # the rerun under the floor
    assert A.angle_label(0.45, FLOOR, 0.1, 1.0, M) == CLOSE  # at the floor, inclusive
    assert A.angle_label(0.9, 0.8, 0.8 - 2 * M + 1e-9, 1.0, M) == TELL  # references do not separate
    assert A.angle_label(0.9, 0.8, 0.8 - 2.5 * M, 1.0, M) == CLOSE  # just separated
    assert A.angle_label(0.8 - M, 0.8, 0.1, 1.0, M) == CLOSE  # a tie with the line
    assert A.angle_label(0.8 - M - 1e-9, 0.8, 0.1, 1.0, M) == FAR
    assert A.angle_label(0.95, 0.8, 0.1, 1.0, M) == CLOSE  # above the rerun
    assert A.angle_label(-0.5, 0.8, 0.1, 1.0, M) == FAR  # a negative cosine


def test_review_examples_and_noise_cannot_give_as_close() -> None:
    assert A.angle_label(0.5, 0.8, 0.1, 1.0, 0.3) == TELL  # read "copies" before the fix
    assert A.angle_label(0.5, 0.8, 0.1, 1.0, 0.05) == FAR
    assert A.angle_label(0.16, 0.2, 0.05, 1.0, 0.05) == TELL  # the old rule's "copies" at a cosine of 0.16
    assert A.angle_label(0.70, 0.80, 0.1, 1.0, 0.05) == FAR  # further, at cosine 0.70: the label says no more than that
    assert A.angle_label(0.78, 0.8, 0.1, 1.0, 0.3) == TELL  # inside the band, with noise over twice the band
    assert A.angle_label(0.78, 0.8, 0.1, 1.0, 2 * M) == CLOSE  # noise at the cap
    assert A.angle_label(0.78, 0.8, 0.1, 1.0, 2 * M + 1e-9) == TELL
    for m in (0.05, 0.1, 0.15, 0.2, 0.3):  # widening the margin never produces the first label
        assert A.angle_label(0.7, 0.8, 0.1, 1.0, m) != CLOSE


def test_ceiling_is_the_as_close_control() -> None:
    assert A.angle_label(0.75, 0.8, 0.1, 0.8 - M, M) == CLOSE  # the line is just reachable
    assert A.angle_label(0.5, 0.8, 0.1, 0.8 - M - 1e-9, M) == TELL  # unreachable, so "further" would be by construction
    assert A.length_label(0.5, 0.5, 1.0, MD) == CLOSE
    best = math.sqrt(1 - 0.7**2)  # 0.714: nothing first-order gets nearer
    assert A.length_label(best, best - MD, 0.7, MD) == CLOSE
    assert A.length_label(best, best - MD - 1e-6, 0.7, MD) == TELL


def test_length_label_edges() -> None:
    for bad in ((NAN, 0.8, 1.0, MD), (0.8, NAN, 1.0, MD), (0.8, 0.8, NAN, MD), (0.8, 0.8, 1.0, NAN)):
        assert A.length_label(*bad) == "unscorable"
    assert A.length_label(0.8 + MD, 0.8, 1.0, MD) == CLOSE  # a tie with the line
    assert A.length_label(0.8 + MD + 1e-9, 0.8, 1.0, MD) == FAR
    assert A.length_label(1.2, 0.8, 1.0, 0.5) == TELL  # between the band and the margin
    assert A.length_label(0.1, 1.0, 1.0, MD) == CLOSE and A.length_label(0.1, 1.0 + 1e-9, 1.0, MD) == TELL  # a rerun no nearer than no change
    assert A.length_label(1.88, 0.89, 1.0, MD) == FAR  # the review's student at 2.3 times the length, against a rerun at the same cosine
    assert A.length_label(0.8, 0.8, 1.0 + 1e-12, MD) == CLOSE  # a ceiling just over 1 from rounding


def test_seed_class_edges() -> None:
    assert A.seed_class(0.5, [], M) == "unscorable"
    assert A.seed_class(NAN, [0.3], M) == "unscorable"
    assert A.seed_class(0.5, [NAN], M) == "unscorable"
    assert A.seed_class(0.5, [0.5], NAN) == "unscorable"
    assert A.seed_class(0.5, [0.4], M) == A.OWN_RUN
    assert A.seed_class(0.5, [0.4], 0.2) == TELL  # the review's example: a gap of 0.1, noise 0.2, was "follows the task only"
    assert A.seed_class(0.5, [0.48], M) == A.TASK_ONLY and A.seed_class(0.5, [0.5], M) == A.TASK_ONLY  # within the band; a tie
    assert A.seed_class(0.5, [0.48], 0.3) == TELL  # within the band, noise over the cap
    assert A.seed_class(0.3, [0.5], M) == A.TASK_ONLY


def test_margin_edges() -> None:
    assert A.margin({"a": [0.01], "b": [0.02]}, M) == M
    assert A.margin({"a": [0.2], "b": [0.02]}, M) == 0.2
    assert A.margin({"a": [0.01], "rerun_spread": [0.3]}, M) == 0.3
    assert math.isnan(A.margin({"a": [NAN], "b": [0.02]}, M))
    assert math.isnan(A.margin({"a": [], "b": []}, M))
    assert A.margin({"a": [0.0]}, MD) == MD


def test_reading() -> None:
    assert A.reading(CLOSE, CLOSE) == CLOSE and A.reading(FAR, FAR) == FAR
    assert A.reading(CLOSE, FAR) == "depends on the inputs"
    assert A.reading(TELL, FAR) == TELL and A.reading(FAR, "mixed") == TELL
    assert A.reading("unscorable", FAR) == "unscorable" and A.reading(TELL, "unscorable") == "unscorable"


def test_no_code_label_says_copies_or_different() -> None:
    src = (EXP / "amend10.py").read_text().lower()
    assert '"copies' not in src and '"different change' not in src and '"reproduces' not in src


def test_arithmetic() -> None:
    g = torch.Generator().manual_seed(0)
    X = torch.randn(4, 5, 7, generator=g, dtype=torch.float64)
    x = torch.randn(200, 7, generator=g, dtype=torch.float64) + 2.0
    sigma, mu = x.T @ x / 200, x.mean(0)
    G = A.gram(X, sigma)
    want = torch.stack([(X @ x.T).transpose(1, 2).reshape(4, 200, -1)[i] for i in range(4)])  # (N, windows, d_out)
    assert torch.allclose(G, torch.einsum("awo,bwo->ab", want, want) / 200)  # tr(A Sigma B^T) = E <A x, B x>
    centred = (x - mu).T @ (x - mu) / 200
    assert torch.allclose(G - A.mean_part(X, mu), A.gram(X, centred))
    assert torch.allclose(A.gram(X, None), A.gram(X, torch.eye(7, dtype=torch.float64)))
    C = A.cosines(G)
    assert torch.allclose(C.diag(), torch.ones(4, dtype=torch.float64))
    Y = torch.cat([X, torch.zeros(1, 5, 7, dtype=torch.float64)])
    Gz = A.gram(Y, sigma)
    Cz = A.cosines(Gz)  # the zero denominator: a zero change reads nan on its row and column
    assert torch.isnan(Cz[4]).all() and torch.isnan(Cz[:, 4]).all() and not torch.isnan(Cz[:4, :4]).any()
    assert torch.isnan(A.cosines(torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))[0]).all()  # negative diagonal
    assert torch.isnan(A.distances(Gz)[:, 4]).all() and torch.isnan(A.sizes(Gz)[:, 4]).all() and torch.isnan(A.sizes(Gz)[4]).all()

    def norm(a: torch.Tensor) -> float:
        return float(((a @ sigma) * a).sum()) ** 0.5

    assert float(A.distances(G)[0, 1]) == pytest.approx(norm(X[0] - X[1]) / norm(X[1]))
    assert float(A.sizes(G)[0, 1]) == pytest.approx(norm(X[0]) / norm(X[1]))
    d, c, s = float(A.distances(G)[0, 1]), float(C[0, 1]), float(A.sizes(G)[0, 1])
    assert d**2 == pytest.approx(s**2 - 2 * s * c + 1)
    assert A.distances(G).diag().abs().max() < 1e-6


def test_power_control_on_a_falling_spectrum() -> None:
    g = torch.Generator().manual_seed(0)
    T = torch.randn(30, 40, generator=g, dtype=torch.float64)
    Q = torch.linalg.qr(torch.randn(40, 40, generator=g, dtype=torch.float64))[0]

    def cos(Z: torch.Tensor, s: Any) -> float:
        return float(A.cosines(A.gram(torch.stack([T, Z]), s))[0, 1])

    # a slowly falling spectrum with a small tail under eps x mean: the old cumulative rule failed such a Sigma
    vals = torch.cat([torch.linspace(3.0, 0.05, 37), torch.full((3,), 1e-5)]).double()
    sig = Q @ torch.diag(vals) @ Q.T
    Z, kd = A.power_change(T, sig, A.CM.excite_eps, A.CM.rho2, g)
    assert kd == pytest.approx(37 / 40)
    assert cos(Z, sig) >= 0.996 and cos(Z, None) <= A.CM.power_frob_max
    iso = torch.eye(40, dtype=torch.float64)  # no direction unexcited: Z is T, the Frobenius side of the control stops the run
    Zi, kdi = A.power_change(T, iso, A.CM.excite_eps, A.CM.rho2, g)
    assert kdi == 1 and cos(Zi, None) > A.CM.power_frob_max
    zero = torch.zeros(40, 40, dtype=torch.float64)  # a zero Sigma: nan, which the stop refuses
    assert math.isnan(cos(A.power_change(T, zero, A.CM.excite_eps, A.CM.rho2, g)[0], zero))


def test_ceiling_bounds_every_first_order_change_and_is_reached() -> None:
    g = torch.Generator().manual_seed(3)
    U, V = torch.randn(12, 3, generator=g, dtype=torch.float64), torch.randn(9, 3, generator=g, dtype=torch.float64)
    x = torch.randn(400, 9, generator=g, dtype=torch.float64) @ torch.diag(torch.linspace(3, 0.2, 9).double())
    S = x.T @ x / 400
    dW = torch.randn(12, 9, generator=g, dtype=torch.float64)
    b = {"a": split.basis(U, V, 1e-6)}
    ce = A.ceilings({"T": {"a": dW}}, b, {"a": S}, ["T"])["T"]
    assert ce == pytest.approx(math.sqrt(1 - split.four_blocks(dW, b["a"], S)["out_out"]))  # the filed block itself
    for _ in range(50):
        F = U @ torch.randn(3, 9, generator=g, dtype=torch.float64) + torch.randn(12, 3, generator=g, dtype=torch.float64) @ V.T
        assert float(A.cosines(A.gram(torch.stack([F, dW]), S))[0, 1]) <= ce + 1e-9
    # the best change: dW without the out-of-reach block. It reads the ceiling, and its distance is sqrt(1 - ceiling^2)
    S2 = split.read_moments(b["a"], S)[1]
    Pu = b["a"].P[:, b["a"].s > 0]
    best = dW - (dW - Pu @ (Pu.T @ dW)) @ S2 @ torch.linalg.inv(S)
    G = A.gram(torch.stack([best, dW]), S)
    assert float(A.cosines(G)[0, 1]) == pytest.approx(ce)
    assert float(A.distances(G)[0, 1]) == pytest.approx(math.sqrt(1 - ce**2))
    assert math.isnan(A.ceilings({"T": {"a": torch.zeros_like(dW)}}, b, {"a": S}, ["T"])["T"])  # zero denominator


def _reads(own: float, rerun: float, other: float, noise: float = 0.0, ceiling: float = 1.0) -> dict[str, Any]:
    """Made-up reads for every set: cosines by kind of pair, distances as at equal length, `noise` between the halves."""
    names = [*A.FINE_TUNES, *A.OWN, A.REAL_O1_R]

    def val(a: str, b: str) -> float:
        if a in A.OWN:
            r = A.OWN[a]
            return own if r == b else own - 0.1 if b in A.TABLE_FT and r in A.TABLE_FT and A.task_of(r) == A.task_of(b) else other
        return 1.0 if a == b else rerun if a in A.FINE_TUNES and b in A.FINE_TUNES and A.task_of(a) == A.task_of(b) else other

    def st(refs: tuple[str, ...]) -> dict[str, Any]:
        cos = {f"{a} | {b}": val(a, b) for a in names for b in refs if b != A.T_REAL_R} | {f"{A.REAL_O1_R} | {A.T_REAL_R}": val(A.REAL_8C[1], A.T_REAL)}
        dist = {k: math.sqrt(2 - 2 * v) for k, v in cos.items()}
        rec = {"whole": cos, "A": {k: v + noise for k, v in cos.items()}, "B": dict(cos), "dist_whole": dist,
               "dist_A": {k: v + noise for k, v in dist.items()}, "dist_B": dict(dist), "size_whole": dict.fromkeys(cos, 1.0),
               "ceiling": {f"{r} | {r}": ceiling for r in refs}}
        return dict.fromkeys(A.FORMS, rec)

    sets = {f"{k}:{t}": st(A.REFS[t]) for k in ("task", "trigger", "taught") for t in A.REFS}
    sets["pile"] = st((*A.FINE_TUNES, A.T_REAL_R))
    return sets


def _cells(r: dict[str, Any], part: str = "whole") -> list[dict[str, float]]:
    return [s["uncentred"][part] for s in r.values()]  # both forms share one dict in `_reads`


def test_score_reads_each_way() -> None:
    s = A.score(_reads(0.78, 0.8, 0.1))
    assert (s["P1_map"]["angle"], s["P2_real"]["angle"]) == (CLOSE, CLOSE)
    assert (s["P1_map"]["length"], s["P2_real"]["length"]) == (CLOSE, CLOSE)
    s = A.score(_reads(0.3, 0.8, 0.1))
    assert (s["P1_map"]["angle"], s["P2_real"]["angle"], s["P1_map"]["length"]) == (FAR, FAR, FAR)
    assert s["P3_seed"]["pile"] == A.OWN_RUN and s["by_input"]["pile"]["uncentred"]["map"]["students"]["S_rand32_s0"]["size"] == 1.0
    s = A.score(_reads(0.75, 0.2, 0.15))
    assert s["P1_map"]["angle"] == TELL  # the rerun under the floor
    s = A.score(_reads(0.3, 0.8, 0.1, noise=0.5))
    assert s["P1_map"]["angle"] == TELL  # noise swallows the gap
    s = A.score(_reads(0.78, 0.8, 0.1, noise=0.15))
    assert s["P1_map"]["angle"] == TELL  # inside the band, but the noise is over the cap: never "as close"
    s = A.score(_reads(0.3, 0.8, 0.1, ceiling=0.5))
    assert s["P1_map"]["angle"] == TELL and s["P2_real"]["angle"] == TELL  # the as-close branch unreachable
    for key in ("P3_seed", "P4_trigger", "P5_taught", "length_reference"):
        assert key in s
    assert "Not about the trigger" in s["P1_map"]["scope"] and "trigger + task" in s["P2_real"]["scope"]


def test_families_are_scored_apart() -> None:
    r = _reads(0.3, 0.8, 0.1)
    for s, d in zip(_cells(r), _cells(r, "dist_whole"), strict=True):
        for a, b in A.rerun_pairs(A.T_REAL):
            if f"{a} | {b}" in s:
                s[f"{a} | {b}"], d[f"{a} | {b}"] = 0.3, math.sqrt(1.4)
    b = A.score(r)
    assert b["P2_real"]["angle"] == TELL  # the real rerun's own cosine fell under the floor
    assert b["P2_real"]["length"] == TELL  # and no nearer than no change
    assert b["P1_map"]["angle"] == FAR  # the map tables' reading does not move


def test_map_table_split_shows_by_table() -> None:
    r = _reads(0.3, 0.8, 0.1)
    for s in _cells(r):
        for st in ("S_rand32_s0", "S_rand32_s1"):
            s[f"{st} | {A.OWN[st]}"] = 0.79
    v = A.score(r)["by_input"]["pile"]["map"]
    assert v["angle"] == "mixed" and v["by_table"]["rand32"]["angle"] == CLOSE and v["by_table"]["rand32b"]["angle"] == FAR


def test_rerun_spread_and_all_cells_enter_the_margin() -> None:
    r = _reads(0.3, 0.8, 0.1)
    for s in _cells(r):
        s["rand32_s2 | rand32_s0"] = 0.6  # one seed pair off by 0.2, in the whole alone
    b = A.score(r)["by_input"]["pile"]["uncentred"]
    assert b["map"]["noise"]["rerun_spread"] == pytest.approx(0.2) and b["map"]["margin"] == pytest.approx(0.2)
    assert b["real"]["margin"] == pytest.approx(M)
    r = _reads(0.3, 0.8, 0.1)
    for s in _cells(r, "A"):
        s[f"{A.T_NULL} | rand32_s0"] = 0.5  # another task's cell, against 0.1 in half B: 0.4
    assert A.score(r)["by_input"]["pile"]["uncentred"]["map"]["noise"]["split_half"] == pytest.approx(0.4)
    for s in _cells(r, "A"):
        s["S_rand32_s1 | rand32_s0"] = 0.9  # the other seed's student, against 0.2: 0.7
    assert A.score(r)["by_input"]["pile"]["uncentred"]["map"]["noise"]["split_half"] == pytest.approx(0.7)


def test_score_missing_state_is_unscorable() -> None:
    r = _reads(0.3, 0.8, 0.1)
    for s in r.values():
        for f in A.FORMS:
            s[f] = {p: {k: v for k, v in x.items() if not k.startswith("S_rand32_s0 ")} for p, x in s[f].items()}
    assert A.score(r)["P1_map"]["angle"] == "unscorable"
    assert A.score(r)["P2_real"]["angle"] == FAR  # the other family stands
    r = _reads(0.3, 0.8, 0.1)
    for s in r.values():
        for f in A.FORMS:
            s[f] = {p: {k: v for k, v in x.items() if A.REAL_RERUNS[1] not in k} for p, x in s[f].items()}
    assert A.score(r)["P2_real"]["angle"] == "unscorable"  # a rerun gone, no line
    empty = A.score({})
    assert empty["P1_map"]["angle"] == "unscorable" and empty["P2_real"]["length"] == "unscorable"
    assert math.isnan(empty["length_reference"]["pile"]["uncentred"][f"long, {A.LONG_STEPS} steps"])


def test_every_cell_a_rule_reads_is_written() -> None:
    """The real path's names: students, runs and references, against what `named` writes per set."""
    names = {*A.FINE_TUNES, *A.OWN, A.T_REAL_R, A.REAL_O1_R}
    for kind in A.KINDS:
        for fts in A.FAMILY.values():
            for r in fts:
                refs = (*A.FINE_TUNES, A.T_REAL_R) if kind == "pile" else A.REFS[A.task_of(r)]
                assert r in refs
                assert all(a in names and b in refs for a, b in A.rerun_pairs(r))
                assert all(x in names for x in (*A.students_of(r), *A.other_tasks(r), *A.other_seed_students(r)))
    assert A.T_REAL_R in A.REFS[A.EMO] and set(A.LONG_PAIR) <= set(A.REFS[A.LONG_TABLE])
    assert A.other_seed_students(A.T_REAL) == [] and len(A.students_of("rand32b_s0")) == 2


class _FakeModel:
    """Two matrices' inputs of rank 3, by the input cache's shape (B, T, d). No weights, no forward."""

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(1)
        self.basis = torch.randn(3, 12, generator=g)
        self.gen = g

    def parameters(self) -> Any:
        return iter([torch.zeros(1)])

    def __call__(self, rows: torch.Tensor, cache_type: str) -> Any:
        assert cache_type == "input"
        x = torch.randn(*rows.shape, 3, generator=self.gen) @ self.basis
        return SimpleNamespace(cache={"a": x, "b": x + 1.0})


def _fake(tmp_path: Path, monkeypatch: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    monkeypatch.setattr(A, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(A, "LARGE", tmp_path / "large")
    g = torch.Generator().manual_seed(2)
    vec = {n: {k: torch.randn(6, 12, generator=g) for k in ("a", "b")} for n in (A.T_REAL, A.T_NULL, "other")}
    bs = split.bases({k: (torch.randn(6, 2, generator=g), torch.randn(12, 2, generator=g)) for k in ("a", "b")})
    return {"controls": {}, "reads": {}, "seconds": {}, "missing": []}, vec, bs


def test_halves_split_by_source_row(tmp_path: Path, monkeypatch: Any) -> None:
    out, _, _ = _fake(tmp_path, monkeypatch)
    rows = torch.zeros(6, 5, dtype=torch.long)
    row_id = torch.tensor([7, 7, 7, 9, 9, 12])  # a row's windows stay together: rows 7 and 12 in A, row 9 in B
    a, b = A.halves(out, _FakeModel(), "pile", rows, row_id, None, 2)
    assert (a[2], b[2]) == (4 * 5, 2 * 5)
    with pytest.raises(SystemExit):
        A.halves(out, _FakeModel(), "pile", rows, torch.full((6,), 7), None, 2)  # one source row: no split-half
    with pytest.raises(SystemExit):
        A.halves(out, _FakeModel(), "pile", rows[:1], torch.tensor([3]), None, 2)  # one window


def test_read_set_on_a_fake_model(tmp_path: Path, monkeypatch: Any) -> None:
    out, vec, bs = _fake(tmp_path, monkeypatch)
    rows, ids, refs = torch.zeros(8, 5, dtype=torch.long), torch.arange(8), (A.T_REAL, "other")
    h = A.halves(out, _FakeModel(), "pile", rows, ids, None, 2)
    A.power_control(out, vec[A.T_REAL], A.sigmas(h)["uncentred"], "pile")
    assert out["controls"]["power pile"]["cos_sigma"] > 0.99 and set(out["controls"]["power pile"]["k_over_d"]) == {"a", "b"}
    assert out["reads"] == {}  # the control writes no read
    task = A.read_set(out, vec, bs, "task:emoticon", h, refs)
    w = out["reads"]["task:emoticon"]["uncentred"]
    assert abs(w["whole"][f"{A.T_REAL} | {A.T_REAL}"] - 1) < 1e-9 and w["dist_whole"][f"{A.T_REAL} | {A.T_REAL}"] < 1e-6
    assert (tmp_path / "large" / "gram_task_emoticon.pt").is_file()
    assert {"whole", "A", "B", "dist_whole", "dist_A", "dist_B", "size_whole", "ceiling"} == set(w)
    assert 0 < w["ceiling"][f"{A.T_REAL} | {A.T_REAL}"] <= 1 and 0 < out["reads"]["task:emoticon"]["top_matrix_share"][A.T_REAL] <= 1
    trigger = A.read_set(out, vec, bs, "trigger:emoticon", A.halves(out, _FakeModel(), "trigger:emoticon", rows, ids, 2, 2), refs)
    grams = {p: {k: task[0][p][k] + trigger[0][p][k] for k in task[0][p]} for p in task[0]}
    sigma = {f: {k: task[1][f][k] + trigger[1][f][k] for k in task[1][f]} for f in A.FORMS}
    A.write_reads(out, vec, bs, "taught:emoticon", grams, sigma, refs)
    i, j = list(vec).index(A.T_REAL), list(vec).index("other")
    for f in A.FORMS:  # the sum of two Grams is the Gram of the sum
        direct = A.cosines(A.pooled(A.gram(torch.stack([v[k].double() for v in vec.values()]), sigma[f][k]) for k in sigma[f]))
        assert out["reads"]["taught:emoticon"][f]["whole"][f"{A.T_REAL} | other"] == pytest.approx(float(direct[i, j]))


def test_write_reads_stops_on_a_change_with_no_energy(tmp_path: Path, monkeypatch: Any) -> None:
    out, vec, bs = _fake(tmp_path, monkeypatch)
    vec["zero"] = {k: torch.zeros(6, 12) for k in ("a", "b")}
    h = A.halves(out, _FakeModel(), "pile", torch.zeros(8, 5, dtype=torch.long), torch.arange(8), None, 2)
    with pytest.raises(SystemExit):
        A.read_set(out, vec, bs, "pile", h, (A.T_REAL,))


def test_one_rerun_stops(tmp_path: Path, monkeypatch: Any) -> None:
    out, vec, _ = _fake(tmp_path, monkeypatch)
    (tmp_path / "large").mkdir()

    def run(dW: dict[str, torch.Tensor], p: float) -> Any:
        return lambda: (dW, {"p_o": p})

    A.one_rerun(out, vec, "same", run(vec[A.T_REAL], 0.999), A.T_REAL, A.CM.t_p_min)  # reproduces it, and is not kept
    assert "same" not in vec
    cases = (("low", A.CM.t_p_min - 1e-9, None), ("nan", NAN, None), ("off", 0.999, A.T_REAL), ("absent", 0.999, "no such file"))
    for name, p, saved in cases:
        with pytest.raises(SystemExit):
            A.one_rerun(out, vec, name, run(vec["other"], p), saved, A.CM.t_p_min)
    A.one_rerun(out, vec, "kept", run(vec["other"], A.CM.t_p_min), None, A.CM.t_p_min)  # inclusive
    assert "kept" in vec and (tmp_path / "large" / "kept_dW.pt").is_file()


def test_filing_matches_the_config() -> None:
    filed = json.loads((EXP / "results" / "predictions_amend10.json").read_text())
    assert filed["amend10"] == json.loads(json.dumps(asdict(A.CM)))
