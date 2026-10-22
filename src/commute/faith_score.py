"""Scoring of Pile-harness readings against a same-job W0, shared by the SGD experiments (sgd_ladder; sgd_faith should
call these instead of its own copies).

Added 2026-10-21. The W0 gate and the trigger-local ratios are the rules filed in
experiments/sgd_faith/results/predictions.json; the three-way verdict is that plan's "not decisive on this card" band.
"""

from __future__ import annotations

from typing import Any

LINE = 1.5  # round 2's adversarial pass line
# W0 under the Pile harness, seed 0, mean of the two blocks: 0.692 / 0.348 (E11 job 25863, L40), 0.691 / 0.348 (E4, 4090)
W0_PGD, W0_PGD_TOL = 0.69, 0.10  # relative
W0_CI, W0_CI_TOL = 0.348, 0.02  # relative; deterministic under Gamma frozen
FLOOR_STOP = 0.25  # W0's PGD floor (max/min - 1 over adversary seeds); 0.209 on the L40 (E8, E11, E19)


def seed_floor(pile: dict[int, dict[str, float]], key: str) -> float:
    """max / min - 1 of W0's `key` over the adversary seeds."""
    v = [p[key] for p in pile.values()]
    return max(v) / min(v) - 1


def w0_check(pile: dict[int, dict[str, float]]) -> dict[str, Any]:
    """W0 at its known Pile values under seed 0, and its PGD floor small enough to read the 1.5 line.
    `pile` = harness.Context.pile (adversary seed -> pile_faithfulness at W0)."""
    w0 = pile[0]
    rec = {"pile": pile, "pgd_floor": seed_floor(pile, "pgd_kl"), "stochastic_floor": seed_floor(pile, "stochastic_kl"),
           "pgd_rel": w0["pgd_kl"] / W0_PGD - 1, "ci_rel": w0["ci_masked_kl"] / W0_CI - 1}
    rec["pass"] = abs(rec["pgd_rel"]) <= W0_PGD_TOL and abs(rec["ci_rel"]) <= W0_CI_TOL and rec["pgd_floor"] <= FLOOR_STOP
    return rec


def verdict(ratio: float, f: float, line: float = LINE) -> str:
    """'pass' at or below line / (1 + f), 'fail' at or above line (1 + f), 'inconclusive' between (f: the same-job W0
    floor; about 1.24 to 1.81 on the L40's f of 0.21)."""
    if ratio <= line / (1 + f):
        return "pass"
    return "fail" if ratio >= line * (1 + f) else "inconclusive"


def trigger_ratios(m: dict[str, Any]) -> dict[str, float]:
    """E4's trigger-local masked checks from a harness.measure record's route-split cells: CI-masked P(o) over target
    P(o) at the trigger, under the state's own gates (L11) and under W0's (L10)."""
    p = m["routes"]["p_cells"]
    return {"state_gated": p["L11"] / p["T1"], "w0_gated": p["L10"] / p["T1"], "target_p_o": p["T1"]}


FLOOR_KEY = {"stochastic_kl": "stochastic_floor", "pgd_kl": "pgd_floor"}  # each Pile reading against its own W0 floor


def state_verdict(pile: dict[str, float], p_o: float, w0: dict[str, Any], p_min: float = 0.9) -> dict[str, Any]:
    """E21's rule for one updated state (sgd_ladder_long's `state_verdict`, moved here 2026-10-22 for
    ladder_second_seed): held-out P(target) below p_min -> 'below P 0.9'; else the stochastic and PGD ratios (`pile`,
    harness ratios to W0) each by `verdict` against its own floor in `w0` (a `w0_check` record): any 'fail' -> fail,
    any 'inconclusive' -> inconclusive, else pass."""
    per = {k: verdict(pile[k], w0[f]) for k, f in FLOOR_KEY.items()}
    overall = "fail" if "fail" in per.values() else "inconclusive" if "inconclusive" in per.values() else "pass"
    return {"pile": per, "verdict": overall if p_o >= p_min else f"below P {p_min:g}"}
