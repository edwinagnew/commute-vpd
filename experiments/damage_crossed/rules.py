"""E122 damage_crossed: the one config and the filed rules (pure; no model). run.py imports both; a CPU test can too.
Plan reports/plans/2026-11-06_plan_e122-damage-crossed-pair.md; filing results/predictions.json."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

SETS = {"A": 0, "B": 256}  # Pile val rows start .. start + 255, two blocks of 128: E39 A3's first and second rows


@dataclass(frozen=True)
class Config:
    # the check: broad_fragility/run.py:66-68, 85-86 (E39); one adversary per 128-row block, batches of 8
    pass_line: float = 1.40
    fail_line: float = 1.61
    pgd_steps: int = 20
    adv_seed: int = 0
    block: int = 128
    # L40 references, mean PGD KL over a set's two blocks at seed 0. W0 and tt_* on A: row_blocks/results/a.json
    # (E49; row_blocks/run.py:71, 75); on B: claim_tests/results/t2c.json (E47) W0.kl.h0, states.*.kl.h0
    w0_ref: dict[str, float] = field(default_factory=lambda: {"A": 0.6917431652545929, "B": 0.6301094591617584})
    saved_ref: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "tt_adam": {"A": 1.5018157362937927, "B": 1.1955474615097046},
        "tt_lora": {"A": 0.9174503982067108, "B": 0.7643491327762604}})
    repro_tol: float = 1e-3  # relative; E47's f_repro_tol, row_blocks/run.py:77
    rerun_tol: float = 1e-6  # W0 reread in a later job: row_blocks/run.py:224 (E47 jobs agreed to 17 digits)
    map_tol: float = 1e-5  # relative: the per-position map's mean against the adversary's own returned KL
    # the fine-tunes: lowercase.py:21, 57, 77 (E37/E39); Adam 3e-5, batch 8, rows once, clip 1; LoRA rank 8 Adam 1e-4 seed 0
    ft_batch: int = 8  # the Adam fine-tune is `lowercase.arm_b_states(..., "adam", seed 0)`: 3e-5, lowercase.py:77
    adam_steps: int = 300
    adam_every: int = 10  # damage curve resolution, Adam
    lora_rank: int = 8
    lora_lrs: tuple[float, ...] = (1e-4, 3e-4)  # E39's chosen rate, then its grid's next (broad_fragility/run.py:82)
    lora_steps: int = 4000  # 32,000 fresh lowercase rows; Pile train 0-32767 make 32,970 (measured, scratch count_rows.out)
    lora_every: int = 100
    lora_source_rows: int = 32768
    gain_ref: dict[str, float] = field(default_factory=lambda: {"adam": 0.1301, "lora": 0.1479})  # row_blocks/run.py:66
    gain_tol: float = 0.005  # row_blocks/run.py:67
    band_frac: float = 0.25  # band half-width w = band_frac x (D_A - D_L), both measured in stage 2
    # the copy: broad_fragility/run.py:61-64, row_copy.copy_on_rows (E37's recipe); E37's 2,400 lowercase rows
    copy_lr: float = 1e-5
    copy_batch: int = 8
    copy_steps: int = 1000
    copy_seed: int = 0
    n_held: int = 64  # held-out lowercase rows and plain rows of damage (broad_fragility/run.py:57, 137-138)
    eval_batch: int = 8
    # time on the L40, seconds (sources in the plan): a check on one set 116, a copy 152, the curves ~470-770
    budget_s: float = 1560.0  # a unit starts only if it ends inside 26 min of the 30-min job
    read_s: float = 135.0
    copy_s: float = 175.0
    curves_s: float = 840.0

    def __post_init__(self) -> None:
        assert self.pass_line < self.fail_line and set(self.w0_ref) == set(SETS) and self.block * 2 == 256
        assert self.adam_steps % self.adam_every == 0 and self.lora_steps % self.lora_every == 0
        assert self.adam_steps % self.lora_every == 0  # the LoRA curve holds step 300, E39's LoRA
        assert self.lora_steps * self.ft_batch <= 32970 and 0 < self.band_frac < 0.5


CFG = Config()
ADAM_CKPTS = tuple(range(CFG.adam_every, CFG.adam_steps + 1, CFG.adam_every))
LORA_CKPTS = tuple(range(CFG.lora_every, CFG.lora_steps + 1, CFG.lora_every))
CELLS = {"c1": "adam", "c2": "lora"}  # cell 1: Adam stopped at the LoRA's damage; cell 2: LoRA trained to Adam's


def finite(*xs: float) -> bool:
    return all(isinstance(x, int | float) and math.isfinite(x) for x in xs)


# ----------------------------------------------------------------------------- rules (pure), as filed


def choose(curve: dict[int, float], target: float, w: float) -> dict[str, Any]:
    """The checkpoint whose damage is nearest `target` (ties: the earliest step); in band if within w of it."""
    pts = sorted((abs(d - target), s, d) for s, d in curve.items() if finite(d))
    if not pts:
        return {"step": None, "damage": None, "in_band": False}
    _, s, d = pts[0]
    return {"step": s, "damage": d, "in_band": abs(d - target) <= w}


def verdict(r: float | None) -> str:
    """E39's lines, both inclusive."""
    if r is None or not finite(r):
        return "invalid" if r is not None else "not run"
    return "pass" if r <= CFG.pass_line else "fail" if r >= CFG.fail_line else "between"


def ratio(kl: dict[str, list[float]], name: str, s: str) -> float | None:
    """A state's mean PGD KL over a set's two blocks over W0's, same seed, same job (E49's `pair`)."""
    a, b = kl.get(f"{name}|{s}"), kl.get(f"W0|{s}")
    return None if a is None or b is None else sum(a) / sum(b)


def cell_verdict(rA: float | None, rB: float | None) -> str:
    """pass: both sets pass; fail: both fail; mixed: anything else read; not run: a set missing; invalid: a non-finite read."""
    v = (verdict(rA), verdict(rB))
    if "not run" in v or "invalid" in v:
        return "not run" if "not run" in v else "invalid"
    return v[0] if v[0] == v[1] and v[0] in ("pass", "fail") else "mixed"


def readable(cell: dict[str, Any]) -> str:
    """'yes', or why the cell's verdict cannot be read: off target (fine-tune or copy damage outside the band), mixed,
    or a pass where its do-nothing does not fail on set A (the check could not fail this update: E49, decisions row 75)."""
    for k, what in (("ft_in_band", "no fine-tune checkpoint in the band"), ("copy_in_band", "the copy's damage is outside the band")):
        if cell.get(k) is None:
            return "not run"
        if not cell[k]:
            return f"off target: {what}"
    v = cell["verdict"]
    if v not in ("pass", "fail"):
        return v
    if v == "pass" and verdict(cell.get("nothing_A")) != "fail":
        return "pass without power: its do-nothing does not fail on set A"
    return "yes"


OUTCOME = {("pass", "fail"): "damage", ("fail", "pass"): "recipe", ("fail", "fail"): "both fail",
           ("pass", "pass"): "both pass"}


def outcome(c1: dict[str, Any], c2: dict[str, Any]) -> dict[str, Any]:
    """The filed reading of the pair. Both readable: OUTCOME. One readable: its half ('c1 pass' leans damage, 'c1 fail'
    leans recipe, 'c2 fail' leans damage, 'c2 pass' leans recipe), no call. Neither: no call."""
    r1, r2 = readable(c1) == "yes", readable(c2) == "yes"
    if r1 and r2:
        return {"call": OUTCOME[(c1["verdict"], c2["verdict"])]}
    half = {("c1", "pass"): "damage", ("c1", "fail"): "recipe", ("c2", "fail"): "damage", ("c2", "pass"): "recipe"}
    if r1 or r2:
        n, c = ("c1", c1) if r1 else ("c2", c2)
        return {"call": "no call", "half": f"{n} {c['verdict']}: leans {half[(n, c['verdict'])]}"}
    return {"call": "no call"}


def case_mask(rows: Tensor, capital: Tensor) -> Tensor:
    """(N, T) bool: position t predicts a next token whose string holds an ASCII capital (`capital`, (V,) bool); the
    last position predicts nothing in the row and is False."""
    m = torch.zeros(rows.shape, dtype=torch.bool)
    m[:, :-1] = capital[rows[:, 1:]]
    return m


def excess_map(m_state: Tensor, m_w0: Tensor, case: Tensor) -> dict[str, Any]:
    """The adversary's excess over W0 per position (state's KL minus W0's, each under its own adversary), split by case
    positions: their share of positions, of the excess, the ratio of the two (1 = spread evenly), and the mean excess
    at case and other positions (nats). No excess (total <= 0): the shares are None."""
    ex, n_case = (m_state - m_w0).double(), int(case.sum())
    tot, p = float(ex.sum()), n_case / case.numel()
    rec: dict[str, Any] = {"case_share_of_positions": p, "mean_excess": float(ex.mean()),
                           "mean_excess_case": float(ex[case].mean()) if n_case else None,
                           "mean_excess_other": float(ex[~case].mean()) if n_case < case.numel() else None}
    share = float(ex[case].sum()) / tot if tot > 0 else None
    rec["case_share_of_excess"] = share
    rec["concentration"] = share / p if share is not None and p > 0 else None
    return rec


def map_contrast(maps: dict[str, dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Is state a's excess more concentrated at case positions than state b's? The difference of their concentrations,
    on each set, against a noise floor = the larger of the two states' own set A - set B differences. 'a more' / 'b more'
    if both sets' differences exceed the floor with one sign, 'no difference' if both lie within it, else 'unclear';
    'not read' if any concentration is missing."""
    got = {(n, s): maps.get(f"{n}|{s}", {}).get("concentration") for n in (a, b) for s in SETS}
    c = {k: float(v) for k, v in got.items() if v is not None}
    if len(c) < len(got):
        return {"reading": "not read"}
    d = {s: c[(a, s)] - c[(b, s)] for s in SETS}
    floor = max(abs(c[(n, "A")] - c[(n, "B")]) for n in (a, b))
    if all(x > floor for x in d.values()) or all(x < -floor for x in d.values()):
        reading = f"{a} more" if d["A"] > 0 else f"{b} more"
    else:
        reading = "no difference" if all(abs(x) <= floor for x in d.values()) else "unclear"
    return {"reading": reading, "difference": d, "floor": floor}


def score(out: dict[str, Any]) -> dict[str, Any]:
    """The filed rules on whatever has been read; a missing read leaves its part 'not run'."""
    kl, sc = out["kl"], {"ratios": {}, "cells": {}}
    for k in kl:
        name, s = k.split("|")
        if name != "W0":
            r = ratio(kl, name, s)
            sc["ratios"][k] = {"ratio": r, "verdict": verdict(r)}
    for c in CELLS:
        ch, cp = out.get("chosen", {}).get(c, {}), out.get("copies", {}).get(f"tt_{c}", {})
        rA, rB = ratio(kl, f"tt_{c}", "A"), ratio(kl, f"tt_{c}", "B")
        cell = {"ft_in_band": ch.get("in_band"), "copy_in_band": cp.get("in_band"),
                "verdict": cell_verdict(rA, rB), "A": rA, "B": rB, "nothing_A": ratio(kl, f"nothing_{c}", "A")}
        cell["readable"] = readable(cell)
        sc["cells"][c] = cell
    sc["outcome"] = outcome(sc["cells"]["c1"], sc["cells"]["c2"])
    maps = out.get("maps", {})
    sc["maps"] = {"M1 tt_adam vs tt_lora": map_contrast(maps, "tt_adam", "tt_lora"),
                  "M2 tt_c1 vs tt_c2": map_contrast(maps, "tt_c1", "tt_c2")}
    return sc
