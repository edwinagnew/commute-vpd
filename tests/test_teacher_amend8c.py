"""CPU check of teacher_training Amendment 8c's pure rules (experiments/teacher_training/amend8c.py) at their edges and of the
filing; no model, no GPU."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1] / "experiments" / "teacher_training"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # the siblings import each other by these names
    spec.loader.exec_module(mod)
    return mod


def _c() -> Any:
    for name in ("run", "newskill", "newskill_amend1", "path", "amend8", "amend8b"):
        _load(name)
    return _load("amend8c")


def test_reproduces_class():
    c = _c()
    assert c.reproduces_class(0.75, 1.0) == ("reproduces", 0.75)  # inclusive
    assert c.reproduces_class(0.25, 1.0) == ("does not", 0.25)  # inclusive
    assert c.reproduces_class(0.5, 1.0)[0] == "partly"
    assert c.reproduces_class(-0.1, 1.0)[0] == "does not"
    assert c.reproduces_class(0.5, 0.2)[0] == "reproduces"  # the control at its line; the ratio is 2.5
    assert c.reproduces_class(0.5, 0.19) == ("no power", None)
    assert c.reproduces_class(math.nan, 1.0) == c.reproduces_class(1.0, math.inf) == ("unscorable", None)


def test_along_class():
    c = _c()
    assert c.along_class(0.09, 0.09) == "neither"
    assert c.along_class(-0.5, -0.5) == "neither"
    assert c.along_class(0.2, 0.1) == "own"  # exactly four times the share
    assert c.along_class(0.1, 0.2) == "other"
    assert c.along_class(0.2, 0.11) == "both"
    assert c.along_class(0.1, 0.1) == "both"  # equal, at the line
    assert c.along_class(0.1, -0.3) == "own"  # an opposite direction has share 0
    assert c.along_class(0.1, 0.0) == "own"
    assert c.along_class(math.nan, 0.1) == "unscorable"
    assert c.share(-0.5) == 0.0 and abs(c.share(0.5) - 0.25) < 1e-12


def test_plane_shared_and_inflation():
    c = _c()
    assert c.plane_share(0.1, 0.1, 1.0) is None and c.plane_share(math.nan, 0.1, 0.0) is None
    assert abs(c.plane_share(0.3, 0.4, 0.0) - 0.25) < 1e-12  # orthogonal teachers: the sum of the shares
    assert abs(c.plane_share(0.5, 0.5, 0.5) - 1 / 3) < 1e-12
    assert c.shared_of(0.5, 0.0, 1.0) is None
    assert abs(c.shared_of(0.09, 1.0, 1.41) - 0.1269) < 1e-12
    assert c.inflation_class((0.1, 0.08), (0.06, 0.06))[0] == "inflated"  # 0.03 above a range of 0.02
    assert c.inflation_class((0.1, 0.06), (0.06, 0.06))[0] == "not shown"  # 0.02 against a range of 0.04
    assert c.inflation_class((0.05, 0.05), (0.05, 0.05))[0] == "not shown"  # a tie
    assert c.inflation_class((math.nan, 0.1), (0.1, 0.1)) == ("unscorable", None)


def test_score_on_a_synthetic_file():
    c = _c()
    cos = {c.key(a, b): 0.05 for a, b in combinations(c.VECTORS, 2)}
    cos |= {c.key(c.R0, c.R1): 0.93, c.key(c.N0, c.N1): 0.6, c.key(c.R1, c.TR): 0.25, c.key(c.N1, c.TN): 0.3, c.key(c.R0, c.N0): 0.09}
    s = c.score(cos, {n: 1.0 for n in c.VECTORS})
    assert s["reproduces"] == "partly" and s["follows_own_teacher"] is True
    assert s["along"][c.R1]["class"] == s["along"][c.N1]["class"] == "own"
    cos[c.key(c.N1, c.TN)] = math.nan
    assert c.score(cos, {n: 1.0 for n in c.VECTORS})["follows_own_teacher"] is None


def test_key_ignores_order_and_filing_matches_the_code():
    c = _c()
    assert c.key(c.TN, c.R0) == c.key(c.R0, c.TN)
    filed = json.loads(c.PRED.read_text())
    assert filed["amend8c"] == json.loads(json.dumps(asdict(c.CM)))
    assert filed["config"] == json.loads(json.dumps(asdict(c.CFG)))
