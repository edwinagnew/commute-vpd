"""CPU check of teacher_training Amendment 9's rules (experiments/teacher_training/amend9.py) at their edges, of its reads on
small random matrices, and of the filing; no model, no saved state. Needs E12's amend1 importable:
PYTHONPATH=experiments/new_vs_existing ~/commute-agents/fleet/cpu-run --mem 8G make test FILES="tests/test_teacher_amend9.py"
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
A = _load("amend9")


def test_along_class_edges():
    line = math.sqrt(0.5)
    assert A.along_class(line + 1e-12, 0.5) == "mostly along"
    assert A.along_class(line - 1e-6, 0.5) == "mostly elsewhere"
    assert A.along_class(-line - 0.1, 0.5) == "not along"  # a large negative cosine has a large square and is not along
    assert A.along_class(0.1, 0.5) == "mostly elsewhere"  # inclusive at along_min
    assert A.along_class(0.0999, 0.5) == "not along"
    assert A.along_class(0.0, 0.5) == "not along"  # a tie at zero
    assert A.along_class(NAN, 0.5) == A.along_class(0.5, NAN) == "unscorable"  # a missing state or a zero denominator
    assert A.along_class(1.0, 0.0) == "unscorable"  # no inside part
    assert A.along_class(1.0, 0.9e-4) == "unscorable"  # an inside part that may be rounding
    assert A.along_class(1.0, 1e-4) == "mostly along"  # inclusive at min_share


def test_wording_edges():
    assert A.wording([]) == "unscorable"
    assert A.wording(["mostly along", "unscorable"]) == "unscorable"
    assert A.wording(["mostly along", "mostly along"]) == "copies"
    assert A.wording(["mostly elsewhere", "not along"]) == "expressible change"
    assert A.wording(["mostly along", "not along"]) == "mixed"


def test_separated_edges():
    assert A.separated([], [0.0]) == A.separated([0.3], []) == A.separated([NAN], [0.0]) == "unscorable"
    assert A.separated([0.3, 0.4], [0.05, 0.06]) == "reads its own"
    assert A.separated([0.3], [0.3]) == "no power"  # a tie
    assert A.separated([0.3], [0.5]) == "no power"
    assert A.separated([0.3], [0.1]) == "partly"  # the other at the line is not under it
    assert A.separated([0.1], [0.05]) == "reads its own"  # own at the line
    assert A.separated([0.08], [0.05]) == "partly"  # own under the line: above the other, not along it
    assert A.separated([0.3], [-0.2]) == "reads its own"


def test_span_class_edges():
    assert A.span_class(0.28, 0.28 / math.sqrt(0.8), 0.8) == "in the span"
    assert A.span_class(1e-3, 0.0, 1.0) == "in the span"  # inclusive at read_tol
    assert A.span_class(0.28 + 2e-3, 0.28 / math.sqrt(0.8), 0.8) == "outside counts"
    assert A.span_class(0.28, 0.28, 1.0) == "in the span"  # every matrix saturated: no outside part at all
    assert A.span_class(0.28, 0.28, 0.0) == "unscorable"
    assert A.span_class(NAN, 0.28, 0.8) == A.span_class(0.28, NAN, 0.8) == A.span_class(0.28, 0.28, NAN) == "unscorable"


def _small(seed: int = 0) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    g = torch.Generator().manual_seed(seed)
    uv = {"a": (torch.randn(12, 3, generator=g), torch.randn(8, 3, generator=g)), "sat": (torch.randn(6, 8, generator=g), torch.randn(6, 8, generator=g))}
    T = {"a": 0.01 * torch.randn(12, 8, generator=g), "sat": 0.01 * torch.randn(6, 6, generator=g)}
    return uv, T, A.split.bases(uv, A.CM.tol)


def test_reads_on_random_matrices():
    uv, T, b = _small()
    I = A.inside(T, b)
    rec = A.control(T, I, uv, b)
    assert not rec["stop"] and 0 < rec["share"] < 1
    assert abs(rec["ls_first_order"] - 1) < 1e-9 and abs(rec["do_nothing"] - math.sqrt(rec["share"])) < 1e-9
    assert rec["ls_whole"] < 1  # the whole state carries dU dV^T, so it is below 1 at any size
    n = A.change_stats(T, T)["cosine"]
    assert abs(n - 1) < 1e-12
    r = A.read(T, T, I)  # the do-nothing's change, read as a student: outside overlap is 1 - share
    assert abs(r["cos_all"] - 1) < 1e-12 and abs(r["out_term"] - (1 - rec["share"])) < 1e-9
    r = A.read(I, T, I)  # an all-inside change: the bound form holds exactly
    assert abs(r["cos_in"] - 1) < 1e-9 and abs(r["cos_all"] - math.sqrt(rec["share"])) < 1e-9 and abs(r["out_term"]) < 1e-9
    other = {k: 0.01 * torch.randn_like(v) for k, v in T.items()}
    a = A.read(A.inside(other, b), T, I)  # a change inside the span in any direction obeys cos_all = cos_in x sqrt(share)
    assert abs(a["out_term"]) < 1e-9 and abs(a["cos_all"]) <= math.sqrt(rec["share"]) + 1e-9
    assert A.span_class(a["cos_all"], a["cos_in"], a["share"]) == "in the span"


def test_zero_change_and_zero_inside():
    _, T, b = _small()
    I = A.inside(T, b)
    r = A.read({k: torch.zeros_like(v) for k, v in T.items()}, T, I)  # W0: no change, so nan, and the rule refuses it
    assert math.isnan(r["cos_all"]) and math.isnan(r["cos_in"]) and A.along_class(r["cos_in"], r["share"]) == "unscorable"
    outside = {k: T[k].double() - I[k] for k in T}  # a fine-tune wholly outside: inside share 0, cosine nan
    I0 = A.inside(outside, b)
    assert A.change_stats(I0, outside)["size_ratio"] < 1e-9


def test_second_order_term_leaves_the_span():
    """The brief's 'only in the span' holds to first order: dU dV^T alone can lie outside."""
    uv, _, _ = _small()
    b = A.split.bases(uv, A.CM.tol)
    g = torch.Generator().manual_seed(1)
    dU, dV = 0.1 * torch.randn(12, 3, generator=g), 0.1 * torch.randn(8, 3, generator=g)
    U, V = uv["a"]
    first = dU @ V.T + U @ dV.T
    assert float(A.split.obstruction(first, b["a"]).norm()) < 1e-5 * float(first.norm())
    assert float(A.split.obstruction(dU @ dV.T, b["a"]).norm()) > 0.1 * float((dU @ dV.T).norm())


def test_score_with_a_missing_state():
    out = {"reads": {}}
    s = A.score(out)
    assert s["P1"]["verdict"] == s["P2"]["null_student"] == s["P2"]["other_table"] == s["P3"]["verdict"] == "unscorable"


def test_filing():
    filed = json.loads(A.PRED.read_text())
    assert filed["amend9"] == json.loads(json.dumps(asdict(A.CM)))
