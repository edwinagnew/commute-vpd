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
M = A.CM.gap_min


def test_table_label_edges() -> None:
    assert A.table_label(NAN, 0.8, 0.1, M) == "unscorable"
    assert A.table_label(0.5, NAN, 0.1, M) == "unscorable"
    assert A.table_label(0.5, 0.8, NAN, M) == "unscorable"
    assert A.table_label(0.5, 0.8, 0.1, NAN) == "unscorable"
    assert A.table_label(0.5, A.CM.along_min - 1e-9, -0.5, M) == "cannot tell"  # the rerun itself is not along
    assert A.table_label(0.9, 0.8, 0.8 - 2 * M + 1e-9, M) == "cannot tell"  # references do not separate
    assert A.table_label(0.9, 0.75, 0.75 - 2.5 * M, M) == "copies"  # just separated
    assert A.table_label(0.8 - M, 0.8, 0.1, M) == "copies"  # a tie with the line
    assert A.table_label(0.8 - M - 1e-9, 0.8, 0.1, M) == "different change"
    assert A.table_label(0.95, 0.8, 0.1, M) == "copies"  # above the rerun
    assert A.table_label(-0.5, 0.8, 0.1, M) == "different change"  # a negative cosine


def test_real_label_edges() -> None:
    band = [0.6, 0.7, 0.8]
    assert A.real_label([], band, 0.1, M) == "unscorable"
    assert A.real_label([0.5], [], 0.1, M) == "unscorable"
    assert A.real_label([0.5, NAN], band, 0.1, M) == "unscorable"
    assert A.real_label([0.5], band, 0.1, NAN) == "unscorable"
    assert A.real_label([0.8 - M, 0.9], band, 0.1, M) == "copies, limited"  # tie, both must clear
    assert A.real_label([0.5, 0.9], band, 0.1, M) == "cannot tell"  # the two orders straddle
    assert A.real_label([0.6 - M - 1e-9], band, 0.1, M) == "different change, limited"
    assert A.real_label([0.6 - M], band, 0.1, M) == "cannot tell"  # a tie is not under the line
    assert A.real_label([0.7], band, 0.1, M) == "cannot tell"  # inside the band
    assert A.real_label([0.2], [0.05, 0.8], -0.5, M) == "cannot tell"  # a rerun that reads nothing
    assert A.real_label([0.2], band, 0.6 - 2 * M + 1e-9, M) == "cannot tell"  # references do not separate
    assert A.real_label([0.7], [0.7], 0.1, M) == "copies, limited"  # one-element band


def test_seed_class_edges() -> None:
    assert A.seed_class(0.5, [], M) == "unscorable"
    assert A.seed_class(NAN, [0.3], M) == "unscorable"
    assert A.seed_class(0.5, [NAN], M) == "unscorable"
    assert A.seed_class(0.75, [0.5], 0.25) == "follows its own run"  # a tie with the line
    assert A.seed_class(0.75, [0.5 + 1e-9], 0.25) == "follows the task only"
    assert A.seed_class(0.3, [0.5], M) == "follows the task only"


def test_margin_edges() -> None:
    assert A.margin([0.01], [0.02], [1e-5]) == M
    assert A.margin([0.2], [0.02], [1e-5]) == 0.2
    assert A.margin([0.01], [0.3], [1e-5]) == 0.3
    assert math.isnan(A.margin([NAN], [0.02], [1e-5]))
    assert math.isnan(A.margin([], [], []))


def test_wording() -> None:
    c, d, cl, dl = "copies", "different change", "copies, limited", "different change, limited"
    assert A.wording(c, c, cl, cl)["wording"].startswith("copies the fine-tune")
    assert "limited" in A.wording(c, c, cl, cl)["scope"]
    assert A.wording(d, d, dl, dl)["wording"].startswith("reproduces its behaviour")
    assert A.wording(d, d, "cannot tell", dl)["scope"].startswith("map tables only")
    assert A.wording(d, d, cl, dl)["wording"] == "cannot tell"  # the real task reads the other way
    assert A.wording(c, d, cl, cl)["wording"] == "depends on the inputs"
    assert A.wording("cannot tell", d, dl, dl)["wording"] == "cannot tell"
    assert A.wording("mixed", "mixed", dl, dl)["wording"] == "cannot tell"
    assert A.wording("unscorable", d, dl, dl)["wording"] == "unscorable"
    assert A.wording(d, d, "unscorable", dl)["wording"] == "unscorable"


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
    Cz = A.cosines(A.gram(Y, sigma))  # the zero denominator: a zero change reads nan on its row and column
    assert torch.isnan(Cz[4]).all() and torch.isnan(Cz[:, 4]).all() and not torch.isnan(Cz[:4, :4]).any()
    assert torch.isnan(A.cosines(torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))[0]).all()  # negative diagonal


def test_power_control_has_power_only_when_sigma_is_low_rank() -> None:
    g = torch.Generator().manual_seed(0)
    T = torch.randn(30, 40, generator=g, dtype=torch.float64)
    Q = torch.linalg.qr(torch.randn(40, 40, generator=g, dtype=torch.float64))[0]
    low = Q @ torch.diag(torch.cat([torch.ones(5), torch.full((35,), 1e-7)]).double()) @ Q.T
    cos = lambda Z, s: float(A.cosines(A.gram(torch.stack([T, Z]), s))[0, 1])
    Z = A.power_change(T, low, A.CM.tail, A.CM.rho2, g)
    assert cos(Z, low) >= A.CM.power_sigma_min and cos(Z, None) <= A.CM.power_frob_max
    iso = torch.eye(40, dtype=torch.float64)  # no direction unexcited: Z is T, the Frobenius side of the control stops the run
    Zi = A.power_change(T, iso, A.CM.tail, A.CM.rho2, g)
    assert cos(Zi, None) > A.CM.power_frob_max


def _reads(own: float, rerun: float, other: float, noise: float = 0.0) -> dict[str, Any]:
    """Made-up reads: every part alike but for `noise` between the halves."""
    def cells(refs: tuple[str, ...], names: list[str]) -> dict[str, float]:
        out = {}
        for b in refs:
            for a in names:
                t = b.split("_")[0]
                out[f"{a} | {b}"] = other
                if a in A.OWN and A.OWN[a] == b or (a in A.REAL_8C and b == A.T_REAL) or (a in A.NULL_8C and b == A.T_NULL):
                    out[f"{a} | {b}"] = own
                elif a in A.FINE_TUNES and a.split("_")[0] == t:
                    out[f"{a} | {b}"] = rerun
                elif a in A.OWN and A.OWN[a].split("_")[0] == t:
                    out[f"{a} | {b}"] = own - 0.1
        out[f"{A.REAL_O1_R} | {A.T_REAL_R}"] = own
        return out

    names = [*A.FINE_TUNES, *A.OWN, *A.REAL_8C, *A.NULL_8C]
    def st(refs: tuple[str, ...]) -> dict[str, Any]:
        w = cells(refs, names)
        return {f: {"whole": w, "A": {k: v + noise for k, v in w.items()}, "B": w} for f in A.FORMS}
    sets = {f"{k}:{t}": st(A.REFS[t]) for k in ("task", "trigger") for t in A.REFS}
    sets["pile"] = st((*A.FINE_TUNES, A.T_REAL_R, A.REAL_8C[1]))
    for s in sets.values():  # the real task's reruns and the band: emoticon's and cipher's own pairs read `rerun`
        for f in A.FORMS:
            for p in A.PARTS:
                s[f][p][f"{A.EMO_RERUN[1]} | {A.EMO_RERUN[0]}"] = rerun
                s[f][p][f"{A.LONG[1]} | {A.LONG[0]}"] = rerun
    return sets


def test_score_reads_both_ways_and_third() -> None:
    assert A.score(_reads(0.78, 0.8, 0.1))["P1_wording"]["wording"].startswith("copies the fine-tune")
    assert A.score(_reads(0.3, 0.8, 0.1))["P1_wording"]["wording"].startswith("reproduces its behaviour")
    assert A.score(_reads(0.75, 0.2, 0.15))["P1_wording"]["wording"] == "cannot tell"  # references do not separate
    assert A.score(_reads(0.3, 0.8, 0.1, noise=0.5))["P1_wording"]["wording"] == "cannot tell"  # noise swallows the gap


def test_score_missing_state_is_unscorable() -> None:
    r = _reads(0.3, 0.8, 0.1)
    for s in r.values():
        for f in A.FORMS:
            for p in A.PARTS:
                s[f][p] = {k: v for k, v in s[f][p].items() if not k.startswith("S_rand32_s0 ")}
    assert A.score(r)["P1_wording"]["wording"] == "unscorable"
    assert A.score({})["P1_wording"]["wording"] == "unscorable"


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


def test_read_set_on_a_fake_model(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(A, "OUT", tmp_path / "out.json")
    monkeypatch.setattr(A, "LARGE", tmp_path / "large")
    g = torch.Generator().manual_seed(2)
    vec = {n: {k: torch.randn(6, 12, generator=g) for k in ("a", "b")} for n in (A.T_REAL, A.T_NULL, "other")}
    vec[A.ZERO] = {k: torch.zeros(6, 12) for k in ("a", "b")}
    out: dict[str, Any] = {"controls": {}, "reads": {}, "concentration": {}, "seconds": {}, "missing": []}
    A.read_set(out, _FakeModel(), vec, "pile", torch.zeros(8, 5, dtype=torch.long), None, 2, (A.T_REAL, A.T_NULL))
    assert out["controls"]["power pile"]["cos_sigma"] > 0.99 and out["controls"]["self pile"]["windows_or_positions"] == 40
    w = out["reads"]["pile"]["uncentred"]["whole"]
    assert abs(w[f"{A.T_REAL} | {A.T_REAL}"] - 1) < 1e-9 and math.isnan(w[f"{A.ZERO} | {A.T_REAL}"])
    assert (tmp_path / "large" / "gram_pile.pt").is_file()
    assert set(out["reads"]["pile"]) == set(A.FORMS) and set(out["reads"]["pile"]["centred"]) == set(A.PARTS)
    A.read_set(out, _FakeModel(), vec, "trigger:emoticon", torch.zeros(9, 5, dtype=torch.long), 2, 2, (A.T_REAL,))
    assert out["controls"]["self trigger:emoticon"]["windows_or_positions"] == 9
    with pytest.raises(SystemExit):
        A.read_set(out, _FakeModel(), vec, "pile", torch.zeros(1, 5, dtype=torch.long), None, 2, (A.T_REAL,))  # one window: no split-half


def test_filing_matches_the_config() -> None:
    filed = json.loads((EXP / "results" / "predictions_amend10.json").read_text())
    assert filed["amend10"] == json.loads(json.dumps(asdict(A.CM)))
