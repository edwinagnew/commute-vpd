"""E78's rules against their edge cases (results/predictions.json .edge_cases), with no model and no GPU."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from commute.jobs import load_experiment

RUN: Any = load_experiment("qwen_anchor1", "run")
P = json.loads(RUN.PRED.read_text())
F = P["files"]
F32 = dict.fromkeys(range(196), "F32")


def test_the_filed_bands_are_e61s() -> None:
    e61 = json.loads((RUN.E61 / "predictions_train.json").read_text())
    want, got = e61["first_minutes"]["a1_1e-6"], P["first_minutes"]
    assert got["stop"] == want["stop"] and got["note"] == want["note"]
    assert P["data"] == e61["data"]["a1"]
    line = json.loads((RUN.E61 / "lines.json").read_text())
    assert P["rule"]["rate_min"] == e61["rules"]["template"]["rate_min"] == line["template_rate_min"]
    assert P["rule"]["strict_min"] == line["template_strict_min"]


def test_the_filed_values_are_e61s_step_100_and_its_own_reads_pass() -> None:
    run = json.loads(RUN.E61_RUN.read_text())
    last = run["steps"][-1]
    R = P["reproduction"]
    assert R["values"] == {"dw_norm": last["dw_norm"], "loss": last["loss"], "held": last["held"], "kl_maths": last["kl"]["maths"]}
    assert RUN.reproduction(last, R)["holds"]
    assert RUN.rule(run["verdict"]["rate"], run["verdict"]["strict"], P["rule"])["learned"]
    bands = P["first_minutes"]
    for severity in ("stop", "note"):
        for at, rows in bands[severity].items():
            for key, band in rows.items():
                assert RUN.in_band(run["steps"][int(at) - 1].get(key), band), (at, key)


def test_a_wrong_state_misses_the_reproduction() -> None:
    fifty = json.loads(RUN.E61_RUN.read_text())["steps"][49]
    rows = RUN.reproduction(fifty, P["reproduction"])["rows"]
    assert not any(r["holds"] for r in rows.values())
    assert not RUN.reproduction({}, P["reproduction"])["holds"]


@pytest.mark.parametrize(("dtypes", "norm", "record", "holds"), [
    (F32, 0.0, 0.0, True),  # step 1: all zeros
    (F32, 1e-9, 0.0, False),  # step 1 not zero
    (F32, 0.4865, 0.4865, True),
    (F32, 0.4866, 0.4865, False),
    (F32, math.nan, 0.4865, False),
    (F32, None, 0.4865, False),
    ({}, None, 0.4865, False),  # no file
    (dict.fromkeys(range(196), "BF16"), 0.4865, 0.4865, False),
    (dict.fromkeys(range(195), "F32"), 0.4865, 0.4865, False),
    ({**F32, 0: "BF16"}, 0.4865, 0.4865, False),
])
def test_file_row(dtypes: dict[int, str], norm: float | None, record: float, holds: bool) -> None:
    assert RUN.file_row(dtypes, norm, record, F)["holds"] is holds


@pytest.mark.parametrize(("aa", "bb", "ab", "holds", "rounding"), [
    (0.25, 0.25 * 0.9999968, 0.25 * 0.9999984 * (1 - 1.39e-6), True, True),  # rounding alone, as measured
    (0.25, 0.25, 0.25 * (1 - 1e-4), True, False),  # a run difference
    (0.2141, 0.2367, math.sqrt(0.2141 * 0.2367) * (1 - 0.0148), False, False),  # E61's step 50 against its step 100
    (0.25, 0.25 * 1.03, 0.25 * math.sqrt(1.03), False, True),  # right direction, wrong size
    (0.0, 0.25, 0.0, False, False),  # a zero file has no cosine
    (0.25, 0.0, 0.0, False, False),
])
def test_against_e61(aa: float, bb: float, ab: float, holds: bool, rounding: bool) -> None:
    got = RUN.against_e61(aa, bb, ab, F)
    assert got["holds"] is holds and got["rounding_alone"] is rounding


@pytest.mark.parametrize(("rate", "strict", "learned"), [
    (0.92578125, 0.62109375, True),
    (0.0, 0.62, False),  # W0's rate
    (0.74, 0.5603162055335968, True),  # on both lines
    (0.92, 0.56, False),
    (0.92, None, False),  # no answer fits the window
    (None, 0.62, False),
])
def test_rule(rate: float | None, strict: float | None, learned: bool) -> None:
    assert RUN.rule(rate, strict, P["rule"])["learned"] is learned


def test_header_and_moments(tmp_path: Path) -> None:
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({"x": torch.tensor([3.0, 0.0]), "y": torch.tensor([[4.0]])}, str(a), metadata={"m": "1"})
    save_file({"x": torch.tensor([0.0, 2.0], dtype=torch.bfloat16), "y": torch.tensor([[1.0]], dtype=torch.bfloat16)}, str(b))
    assert RUN.header(a) == {"x": "F32", "y": "F32"} and RUN.header(b) == {"x": "BF16", "y": "BF16"}
    assert RUN.header(tmp_path / "none.safetensors") == {}
    assert RUN.moments(a, b) == (25.0, 5.0, 4.0)


def test_step_gaps() -> None:
    mine = [{"step": 1, "loss": 1.0, "dw_norm": 0.0}, {"step": 2, "loss": 1.02, "dw_norm": 0.2, "held": 0.5}]
    e61 = [{"step": 1, "loss": 1.0, "dw_norm": 0.0}, {"step": 2, "loss": 1.0, "dw_norm": 0.2, "held": None}]
    got = RUN.step_gaps(mine, e61)
    assert got["loss"]["at_step"] == 2 and got["loss"]["max_rel"] == pytest.approx(0.02) and got["held"] is None
    assert got["dw_norm"]["n"] == 1
