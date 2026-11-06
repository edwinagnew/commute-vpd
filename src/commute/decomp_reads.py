"""The decomposition side's three reads of a saved full-weight update delta-W on a trigger task, for E125
(experiments/fresh_reuse_pair). Moved, with the same calls, from experiments/teacher_route_split/run.py (E119: `label`,
`e12_control`, `Job.read`, the student step) and experiments/recruitment_test/amend2.py (E38 Amendment 2's trigger-only
removal grid, the 'trigger' rule only); both files stay as they ran.

1. The route split (`local_faith.route_split`) of a state, labelled by E12's filed lines as E119 labels them.
2. Teacher training (E34's arm-A student, `teacher.train_student`): the alive U, V trained to copy the state with Gamma
   and Delta frozen, read by the route split on its own state (target W0 + the student's change, its U', V').
3. E38 Amendment 2: S = the alive subcomponents whose mean trigger gate over the training windows rises by more than
   `thr` (E18's THR 0.3), removed at the trigger only from the state and from W0; D(X) = rho_ft(X) - rho_0(X) against
   `n_random` size-matched random sets (`recruit.random_like`), classed by `recruitment.d_seed_class`.

A per-window target uses data.LAST with the windows' 42nd column; every Gamma read gets the first ci_len tokens.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import recruit as R
from commute import recruitment as RT
from commute.faithfulness import model_state
from commute.local_faith import MIN_GAIN, route_split, trigger_mean_gates

Lines = Mapping[str, float]  # r_hi, r_lo, masked_hi, masked_lo, min_ratio (E119's config)


def route_label(r: float, e_gate: float, gain: float, teacher_gain: float, lines: Lines) -> str:
    """E119's `label`: E12's filed lines: re-switching r >= r_hi with e_gate <= masked_lo; masked-off e_gate >= masked_hi
    with r <= r_lo; else neither. Unreadable: a share not finite, gain under MIN_GAIN, or gain / teacher's gain under
    min_ratio (a zero or negative teacher gain is unreadable)."""
    ratio = gain / teacher_gain if teacher_gain > 0 else math.nan
    if not (math.isfinite(r) and math.isfinite(e_gate) and gain >= MIN_GAIN and ratio >= lines["min_ratio"]):
        return "unreadable"
    if r >= lines["r_hi"] and e_gate <= lines["masked_lo"]:
        return "re-switching"
    if e_gate >= lines["masked_hi"] and r <= lines["r_lo"]:
        return "masked-off"
    return "neither"


def e12_control(path: Path, lines: Lines) -> dict[str, dict[str, Any]]:
    """E119's positive control: E12 Amendment 1's batch-order-0 students (state a) under `route_label`, with their class
    (copy tables should read re-switching, maps masked-off)."""
    out = {}
    for n, s in json.loads(path.read_text())["students"].items():
        if "_o" not in n:
            sh = s["a"]["shares"]
            out[n] = {"class": "copy" if n.startswith("ident") else "map", "r": sh["r"], "e_gate": sh["e_gate"],
                      "label": route_label(sh["r"], sh["e_gate"], s["a"]["gain"], s["teacher_gain"], lines)}
    return out


def read_split(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], comps: Any, windows: Tensor, half: int, token: int | Tensor,
               base: dict[str, Any], comps0: dict[str, tuple[Tensor, Tensor]], ci_len: int | None) -> dict[str, Any]:  # fmt: skip
    """E119's `Job.read`: the route split at target W0 + dW with components `comps` (None: released)."""
    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) for k in W0}, components=comps):
        return route_split(model, windows, half, token, base, comps0, ci_len=ci_len)


def switched_set(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], train: Tensor, half: int, m0: dict[str, Tensor],
                 alive: dict[str, Tensor], thr: float) -> tuple[R.Sel, dict[str, Tensor]]:  # fmt: skip
    """E38's S and the shift s it is cut from: s = the state's mean trigger gates over `train` (tokens without a target
    column) minus W0's m0, 0 on dead subcomponents (`regating.shift_of`); S = alive with s > thr (`recruit.switched`)."""
    with model_state(model, target={k: W0[k] + dW[k].to(W0[k].device) for k in W0}):
        m1 = trigger_mean_gates(model, train, half)
    shift = R.mean_change(m1, m0, alive)
    return R.switched(shift, thr), shift


def removal_d(model: Any, W0: dict[str, Tensor], dW: dict[str, Tensor], comps: dict[str, tuple[Tensor, Tensor]], S: R.Sel,
              alive: dict[str, Tensor], windows: Tensor, half: int, token: int, n_random: int, min_gain: float) -> dict[str, Any]:  # fmt: skip
    """E38 Amendment 2's stage-1 read of one state, trigger-only rule: G = T1 - T0; for X in S and n_random random sets,
    rho_ft(X) = (T1 - lo(W1 without X)) / G, rho_0(X) = (T0 - lo(W0 without X)) / G, D = rho_ft - rho_0 (nan when
    G < min_gain); `empty` = lo(W1 without nothing) - T1 (E38 A2's C3, expected 0); the class by d_seed_class, or
    "unscored" when G < min_gain or S is empty (no set to remove: the rule cannot read it, as E38's lowercase)."""
    W1 = {k: W0[k] + dW[k].to(W0[k].device) for k in W0}
    refs, T = {}, {}
    for lab, W in (("0", W0), ("ft", W1)):
        with model_state(model, target=W):
            refs[lab] = RT.window_logits(model, windows)
            T[lab] = RT.readout(model, windows, half, token, refs[lab])["lo"]
    G = T["ft"] - T["0"]
    g = G if G >= min_gain else 0.0
    empty = {k: torch.zeros(0, dtype=torch.long) for k in alive}
    sets = [S, *(R.random_like(S, alive, i) for i in range(n_random))]
    rho = {lab: [RT.share(T[lab], RT.removed_at(model, W, comps, X, windows, half, token, refs[lab])["lo"], g) for X in sets]
           for lab, W in (("0", W0), ("ft", W1))}  # fmt: skip
    D = [a - b for a, b in zip(rho["ft"], rho["0"], strict=True)]
    rec = {"T0": T["0"], "T1": T["ft"], "gain": G, "scorable": G >= min_gain, "n_S": R.size(S),
           "empty": RT.removed_at(model, W1, comps, empty, windows, half, token, refs["ft"])["lo"] - T["ft"],
           "rho_ft": {"S": rho["ft"][0], "random": rho["ft"][1:]}, "rho_0": {"S": rho["0"][0], "random": rho["0"][1:]},
           "D": {"S": D[0], "random": D[1:]}}
    rec["class"] = "unscored: S is empty" if not R.size(S) else RT.d_seed_class(D[0], D[1:]) if rec["scorable"] else RT.UNSCORED_GAIN
    del refs
    return rec
