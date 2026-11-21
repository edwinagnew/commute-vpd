"""Descriptive, $0, saved readings only: two readings of one new idea, "is the answer's gain first order in the update?"

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/new_proxies.py

The idea: a change that turns up a pathway W0 already runs moves the answer in proportion to the change (odd in dW:
-dW takes the answer down by about what +dW added); a pathway built from changed parts that compose with each other
moves it as a product of the changes (even: -dW also raises it, and the gain starts as alpha^2). Two saved readings:

- elasticity: d log G / d log c between E237's saved and scaled states (G = answer log-odds gain, c = E237's scale).
  1 = linear, 2 = a product of two changes, under 1 = saturation. Copy has c = 1, so it is undefined there.
- antisymmetry: -G(W0 - dW) / G(W0 + dW) on E46's states (stage0b). 1 = odd, -1 = even, 0 = no first-order part.

Not filed proxies; kept apart from analyse.py's table. Writes results/new_proxies.json.
"""

from __future__ import annotations

import json
import math
from statistics import fmean

from commute.jobs import load_experiment

A = load_experiment("proxies_vs_r", "analyse")
OUT = A.HERE / "results" / "new_proxies.json"


def elasticity() -> dict[str, float | None]:
    states = A.load("key_grid/results/stage1.json")["states"]
    per: dict[str, list[float]] = {t: [] for t in A.KEY}
    for s in states.values():
        if abs(s["c"] - 1) > 1e-9:
            per[s["task"]].append(math.log(s["scaled"]["gain"] / s["saved"]["gain"]) / math.log(s["c"]))
    return {t: fmean(v) if v else None for t, v in per.items()}


def antisymmetry() -> dict[str, float]:
    st = A.load("regating_table/results/stage0b.json")["states"]
    return {t: -st[f"{t}/flipped"]["G"] / st[f"{t}/s0"]["G"] for t in
            ("ident32", "ident32b", "rand32", "rand32b", "emoticon", "r1_t534_v369", "r2_t359_v816")}  # fmt: skip


def main() -> int:
    r = A.r_size_removed()
    prior = A.proxies()["prior"]
    el = elasticity()
    an = antisymmetry()
    pair = "e187_e46_copy_random"
    out = {
        "note": "descriptive, saved readings only, not filed; see the docstring",
        "elasticity_key": {"values": el, "vs_R": A.spearman(el, r["key"]), "vs_prior": A.spearman(el, prior["key"]),
                           "R": r["key"], "prior": prior["key"]},
        "antisymmetry_e46": {"values": an,
                             "pair_gap_reuse_minus_new": fmean([an["ident32"], an["ident32b"]]) - fmean([an["rand32"], an["rand32b"]]),
                             "R_gap": r["pairs"].get(pair), "prior_gap": prior["pairs"].get(pair)},
    }  # fmt: skip
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
