"""CPU checks of E38's pure readings: the stage-1 decision tree (recruitment.readings), the null-subset draw, and the
census's enrichment and classes (census.enriched, census.classify)."""

from __future__ import annotations

import torch

from commute import census as CE
from commute import recruitment as RT


def rec(rho: float, rnd: float = 0.02, null: float = 0.01, w0: float = 0.0, kl_s: float = 0.1, kl_busy: float = 1.0,
        scorable: bool = True) -> dict:
    return {"rho_S": rho, "random_max": rnd, "rho_null": null, "rho_S_W0": w0, "kl_ft": 0.05, "scorable": scorable,
            "kl_other": {"S": kl_s, "busy": kl_busy}}


def tasks(*rhos: tuple[float, float], **kw) -> dict:
    return {f"t{i}": {"0": rec(a, **kw), "1": rec(b, **kw)} for i, (a, b) in enumerate(rhos)}


def test_pass_partial_fail_outside():
    assert RT.readings(tasks((0.6, 0.65), (0.55, 0.6), (0.1, 0.1)))["reading"] == "pass"
    assert RT.readings(tasks((0.3, 0.35), (0.4, 0.45), (0.6, 0.6)))["reading"] == "partial"
    assert RT.readings(tasks((0.1, 0.1), (0.15, 0.1), (0.9, 0.9)))["reading"] == "fail"
    assert RT.readings(tasks((0.6, 0.6), (0.3, 0.3), (0.1, 0.1)))["reading"] == "outside the tree"


def test_seeds_must_agree():
    out = RT.readings(tasks((0.45, 0.55), (0.1, 0.3), (0.6, 0.6)))  # partial/pass and fail/partial: no class
    assert out["task_class"] == {"t0": "seeds disagree", "t1": "seeds disagree", "t2": "pass"}
    assert out["reading"] == "outside the tree"
    assert RT.readings(tasks((0.5, 0.7), (0.5, 0.7), (0.6, 0.6)))["reading"] != "pass"  # P1's within-0.1 rule
    assert RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.6, 0.6), null=0.2))["reading"] != "pass"


def test_random_bites_and_gain_are_unscored():
    out = RT.readings(tasks((0.1, 0.1), (0.1, 0.1), (0.1, 0.1), rnd=0.2))
    assert set(out["task_class"].values()) == {RT.UNSCORED_RANDOM} and out["reading"] == "outside the tree"
    st = tasks((0.6, 0.6), (0.6, 0.6), (0.1, 0.1))
    st["t0"]["1"] = rec(float("nan"), rnd=float("nan"), null=float("nan"), w0=float("nan"), scorable=False)
    out = RT.readings(st)
    assert out["task_class"]["t0"] == RT.UNSCORED_GAIN and out["reading"] == "outside the tree"
    assert "t0/s1" not in out["P3"]["per_state"] and out["P3"]["unscored"] == ["t0/s1"]


def test_p2_split():
    out = RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.1, 0.1), kl_s=0.2, w0=0.15))
    assert out["P2_S"]["per_task"] == {"t0": False, "t1": False} and out["P2_S_fails_while_P1_passes"]
    assert out["P2_busy"]["pass"] is True and out["P3"]["pass"] is False
    out = RT.readings(tasks((0.6, 0.6), (0.6, 0.6), (0.1, 0.1), kl_busy=0.1))
    assert out["P2_busy"]["pass"] is False and not out["P2_S_fails_while_P1_passes"]
    assert RT.readings(tasks((0.1, 0.1), (0.1, 0.1), (0.1, 0.1)))["P2_S"]["pass"] == "unscored"


def test_subset_like():
    sel = {"a": torch.tensor([0, 1]), "b": torch.tensor([3, 4, 5])}
    pool = {"a": torch.tensor([7, 8, 9]), "b": torch.tensor([2])}
    sub = RT.subset_like(sel, pool, 0)
    assert sub["a"].numel() == 2 and set(sub["a"].tolist()) <= {7, 8, 9} and sub["b"].tolist() == [2]


def test_enriched_and_classify():
    base = torch.tensor([1000, 100, 10, 10])  # token counts over 1120 positions
    on = torch.tensor([10, 50, 6, 1])  # at 67 on-positions
    top = CE.enriched(on, 67, base, 1120, top=3)
    assert [t for t, _, _ in top] == [2, 1, 0]  # token 3 has one occurrence there: below MIN_COUNT, dropped
    row = {"n_on": 67, "top_next": top, "top_cur": [(7, 9.0, 20)]}
    assert CE.classify(row, 1, {7}) == "both" and CE.classify(row, 3, {8}) == "neither"
    assert CE.classify({**row, "n_on": 10}, 1, {7}) == "rare"
