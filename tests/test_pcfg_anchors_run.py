"""E55 run.py's a1c/a2 plumbing on hand-made numbers (review reports/reviews/2026-10-29_review_e55-a1c-a2-code.md):
a2 refuses to start while the bars are the placeholder, a filing must carry both l*, the A2 bars use the pairs built
(E14) and never the last block (E12), and state files are keyed by the arm's recipe. run.py is loaded from its path
(it is a script, not a package module). No model is built or loaded."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RUN = Path(__file__).resolve().parents[1] / "experiments" / "pcfg_anchors" / "run.py"
CHECK_BAR = {"n0": 0.0, "s": 0.001, "p_plus": 0.9, "lo": 0.09, "hi": 0.45, "void": False}


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("pcfg_anchors_run", RUN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    spec.loader.exec_module(mod)
    return mod


def filing(**extra: object) -> dict:
    return {k: dict(CHECK_BAR) for k in ("B_a", "P_a", "B_b", "P_b")} | extra


def test_filed_bars_placeholder_is_none(run):
    assert run.filed_bars({"status": "to be filed"}, {"l_star_a": 3, "l_star_b": 4}) is None


def test_filed_bars_need_both_l_star_equal_to_a1c_patch(run):
    computed = {"l_star_a": 3, "l_star_b": 4}
    assert run.filed_bars(filing(l_star_a=3, l_star_b=4), computed)["l_star_b"] == 4
    with pytest.raises(AssertionError):
        run.filed_bars(filing(l_star_a=3), computed)  # l_star_b missing
    with pytest.raises(AssertionError):
        run.filed_bars(filing(l_star_a=3, l_star_b=5), computed)


def test_a2_refuses_while_bars_are_the_placeholder(run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "predictions.json").write_text(json.dumps({"a2_bars": {"status": "to be filed"}}))
    (tmp_path / "a1c_patch.json").write_text(json.dumps({"a2_bars": {"l_star_a": 3, "l_star_b": 4}}))
    monkeypatch.setattr(run, "PRED", tmp_path / "predictions.json")
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    assert run.stage_a2(["N1"], 1.0) == 2
    assert not (tmp_path / "a2_N1.json").exists()


def test_a2_bars_use_the_fewest_pairs_and_skip_the_last_block(run):
    block_p = [0.1, 0.5, 0.86, 0.9, 0.9, 1.0]  # last block ~1 (output copy); block 3 within 3 s only at n = 500
    patch = {"P_a_positive": [block_p] * 3, "P_b_positive": [block_p] * 3, "P_a_null": [[0.0] * 6] * 3,
             "P_b_null": [[0.0] * 6] * 3, "n_pairs_a": [1000, 500, 1000], "n_pairs_b": [1000] * 3}  # fmt: skip
    b_null = {"B_a_null": [0.0] * 3, "B_b_null": [0.26] * 3}
    rows = {"LOW": {"count_a": 0.99}, "HIGH": {"count_b": 0.99}}
    bars = run.a2_bars(patch, b_null, rows)
    assert (bars["l_star_a"], bars["l_star_b"]) == (3, 4)
    assert bars["P_a"]["s"] == pytest.approx(np.sqrt(0.002 * 0.998 / 500))  # E14: the fewest pairs built
    assert bars["P_b"]["s"] == pytest.approx(np.sqrt(0.001 * 0.999 / 1000))
    assert bars["P_a"]["p_plus"] == pytest.approx(0.86) and bars["P_a"]["lo"] == pytest.approx(0.086)  # E13


def test_state_key_changes_with_every_recipe_field(run, monkeypatch: pytest.MonkeyPatch):
    arm = run.ARMS["N1"]
    key = run.state_key("N1", arm)
    assert key.startswith("N1-") and key == run.state_key("N1", arm)
    changed = [dataclasses.replace(arm, lr=2e-4), dataclasses.replace(arm, c_tr=0.5),
               dataclasses.replace(arm, prior="HIGH"), dataclasses.replace(arm, seed=1)]  # fmt: skip
    keys = {run.state_key("N1", a) for a in changed}
    monkeypatch.setattr(run, "FT_ITERS", 5000)
    keys.add(run.state_key("N1", arm))
    monkeypatch.setattr(run, "FT_BATCH", 64)
    keys.add(run.state_key("N1", arm))
    assert key not in keys and len(keys) == 6
    assert run.state_path(key, 1000).name == f"{key}_1000.pt"
