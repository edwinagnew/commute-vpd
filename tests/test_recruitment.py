"""CPU checks of E38's pure readings: the stage-1 decision tree (recruitment.readings) and the census's enrichment and
classes (census.enriched, census.classify)."""

from __future__ import annotations

import torch

from commute import census as CE
from commute import recruitment as RT


def rec(rho: float, rnd: float = 0.02, null: float = 0.01, w0: float = 0.0, kl_s: float = 0.1, kl_busy: float = 1.0) -> dict:
    return {"rho_S": rho, "random_max": rnd, "rho_null": null, "rho_S_W0": w0, "kl_ft": 0.05,
            "kl_other": {"S": kl_s, "busy": kl_busy}}


def tasks(*rhos: tuple[float, float], **kw: float) -> dict:
    return {f"t{i}": {"0": rec(a, **kw), "1": rec(b, **kw)} for i, (a, b) in enumerate(rhos)}


def test_pass_partial_fail_outside():
    assert RT.readings(tasks((0.6, 0.65), (0.55, 0.6), (0.1, 0.1)))["reading"] == "pass"
    assert RT.readings(tasks((0.3, 0.35), (0.4, 0.45), (0.6, 0.6)))["reading"] == "partial"
    assert RT.readings(tasks((0.1, 0.1), (0.15, 0.1), (0.9, 0.9)))["reading"] == "fail"
    assert RT.readings(tasks((0.6, 0.6), (0.3, 0.3), (0.1, 0.1)))["reading"] == "outside the tree"


def test_p1_needs_seed_agreement_and_low_nulls():
    assert RT.readings(tasks((0.5, 0.7), (0.5, 0.7), (0.6, 0.6)))["reading"] != "pass"  # seeds 0.2 apart on two tasks
    assert RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.6, 0.6), null=0.2))["reading"] != "pass"


def test_fail_when_close_to_random():
    out = RT.readings(tasks((0.3, 0.3), (0.3, 0.3), (0.3, 0.3), rnd=0.25))
    assert out["reading"] == "fail" and all(out["task_fails"].values())


def test_p2_read_only_where_p1_holds_and_p3():
    out = RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.1, 0.1), kl_s=0.2, w0=0.15))
    assert out["P2"]["per_task"] == {"t0": False, "t1": False} and out["P2_fails_while_P1_passes"]
    assert out["P3"]["pass"] is False
    assert RT.readings(tasks((0.1, 0.1), (0.1, 0.1), (0.1, 0.1)))["P2"]["pass"] == "unscored"


def test_enriched_and_classify():
    base = torch.tensor([1000, 100, 10, 10])  # token counts over 1120 positions
    on = torch.tensor([10, 50, 6, 1])  # at 67 on-positions
    top = CE.enriched(on, 67, base, 1120, top=3)
    assert [t for t, _, _ in top] == [2, 1, 0]  # token 3 has one occurrence there: below MIN_COUNT, dropped
    row = {"n_on": 67, "top_next": top, "top_cur": [(7, 9.0, 20)]}
    assert CE.classify(row, 1, {7}) == "both" and CE.classify(row, 3, {8}) == "neither"
    assert CE.classify({**row, "n_on": 10}, 1, {7}) == "rare"
