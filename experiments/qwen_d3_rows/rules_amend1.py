"""E94 Amendment 1: the known-facts states (E103 Amendments 1 and 2, baseline-lead), their numbers beside E94's anchors,
and the filed rule (CPU, no model). Plan: reports/plans/2026-11-05_plan_e94-d3-rows.md, "Amendment 1"; rules:
results/predictions_amend1.json; the run: amend1.py.

  python experiments/qwen_d3_rows/rules_amend1.py           # results/table_amend1.json from whatever is written
  python experiments/qwen_d3_rows/rules_amend1.py --edges   # the plan's edge cases through `side`, before filing

The rule (`side`), per reading and size: the known-facts arm's seed mean placed on the line from anchor 1's mean (f = 0)
to anchor 2's (f = 1). It reads "like anchor 2" when f > 0.5 + noise, "like anchor 1" when f < 0.5 - noise, where noise
is the largest of the three arms' seed spreads over the anchors' gap; "no call" when the anchors do not separate
(`qc.separates`), "unread" when any value is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

import rules

from commute import qwen_comparators as qc
from commute.jobs import EXPERIMENTS

E103 = EXPERIMENTS / "known_answer_pair" / "results"  # baseline-lead's E103, read only
DW, FORMS, SENTENCES = E103 / "large" / "dW", E103 / "forms.json", E103 / "forms_amend1.jsonl"
KNOWN = {"known_s42_matched": "amend1_matched", "known_s142_matched": "amend2_s142_matched",
         "known_s42_end": "amend1_end", "known_s142_end": "amend2_s142_end"}  # name -> file stem in DW
OWN = ("real", "tofu")  # the known-facts arm's own rows, then anchor 2's TOFU rows
SIZES = {"matched": (("known_s42_matched", "known_s142_matched"), rules.PAIRS["matched"]),
         "end": (("known_s42_end", "known_s142_end"), rules.PAIRS["full"])}  # anchor 1's pair is its full = matched state
READINGS = ("outside", "placement", "think")
MIDWAY = "midway: no call"


def path(name: str) -> Any:
    return DW / f"{KNOWN[name]}.safetensors"


# ----------------------------------------------------------------------------- the rule


def side(known: Sequence[float | None], a1: Sequence[float | None], a2: Sequence[float | None]) -> dict[str, Any]:
    """Where the known-facts arm reads between the anchors (module docstring), with both `qc.separates` verdicts beside."""
    out: dict[str, Any] = {"known": known, "anchor1": a1, "anchor2": a2, "anchors": qc.separates(a1, a2),
                           "vs_anchor1": qc.separates(known, a1), "vs_anchor2": qc.separates(known, a2)}
    if "unread" in (out["anchors"], out["vs_anchor1"]):
        return out | {"verdict": "unread"}
    if out["anchors"] != "counts":
        return out | {"verdict": "no call: the anchors do not separate"}
    k, x, y = ([float(v) for v in p if v is not None] for p in (known, a1, a2))
    m1, m2 = sum(x) / 2, sum(y) / 2  # unequal: the anchors' gap is over a spread >= 0
    f = (sum(k) / 2 - m1) / (m2 - m1)
    noise = max(abs(k[0] - k[1]), abs(x[0] - x[1]), abs(y[0] - y[1])) / abs(m2 - m1)
    v = "like anchor 2" if f > 0.5 + noise else "like anchor 1" if f < 0.5 - noise else MIDWAY
    beyond = v != MIDWAY and (f > 1 + noise or f < -noise)
    return out | {"f": f, "noise": noise, "verdict": v + (", beyond it" if beyond else "")}


# ----------------------------------------------------------------------------- numbers and the table


def numbers_of(names: list[str] | tuple[str, ...], own: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """E94's numbers per written state, with the direct share and credit on each of `own[name]`'s row sets beside."""
    out: dict[str, Any] = {}
    for n in names:
        rec = rules.load(n)
        if rec is not None:
            out[n] = rules.numbers(rec, None) | {"own_rows": {r: {k: v for k, v in rules.numbers(rec, r).items() if k in ("direct", "credit")}
                                                              for r in own[n] if r in rec["rows"]}}
    return out


def sides(nums: dict[str, Any]) -> dict[str, Any]:
    get = lambda pair, k: [rules.value(nums.get(n), k) for n in pair]
    return {size: {k: side(get(kn, k), get(rules.PAIRS["a1"], k), get(a2, k)) for k in READINGS} for size, (kn, a2) in SIZES.items()}


def predictions(sd: dict[str, Any], P: dict[str, Any]) -> dict[str, str]:
    """A lean holds when the verdict starts with it; no lean: "reported"."""
    out = {}
    for pid, p in P["predictions"].items():
        v = sd[p["size"]][p["reading"]]["verdict"]
        out[pid] = "unread" if v == "unread" else "reported" if p["lean"] is None else "holds" if v.startswith(p["lean"]) else "misses"
    return out


def score() -> dict[str, Any]:
    anchors = [n for k in ("a1", "matched", "full") for n in rules.PAIRS[k]]
    nums = numbers_of(list(KNOWN), dict.fromkeys(KNOWN, OWN)) | numbers_of(anchors, {n: (rules.STATES[n][2],) for n in anchors})
    sd = sides(nums)
    table = {"numbers": nums, "sides": sd, "predictions": predictions(sd, json.loads((rules.RESULTS / "predictions_amend1.json").read_text()))}
    tmp = rules.RESULTS / "table_amend1.json.tmp"
    tmp.write_text(json.dumps(table, indent=1, default=float) + "\n")
    tmp.replace(rules.RESULTS / "table_amend1.json")
    return table


# ----------------------------------------------------------------------------- the plan's edge cases


def edges() -> list[dict[str, Any]]:
    """Each case of the plan's table, the verdict it names, and what `side` gives; E94's own values from table.json."""
    e94 = json.loads((rules.RESULTS / "table.json").read_text())["numbers"]
    e = lambda pair, k: [e94[n][k]["value"] for n in pair]
    a1o, a2o = e(rules.PAIRS["a1"], "outside"), e(rules.PAIRS["matched"], "outside")
    a1p, a2p = e(rules.PAIRS["a1"], "placement"), e(rules.PAIRS["matched"], "placement")
    zero, one = [0.0, 0.0], [1.0, 1.0]
    cases = [
        ("Tie: f exactly 0.5", side([0.5, 0.5], zero, one), MIDWAY),
        ("f exactly 0.5 + noise (strict)", side([0.625, 0.875], zero, one), MIDWAY),
        ("Both known seeds between, nearer anchor 1", side([0.3, 0.35], zero, one), "like anchor 1"),
        ("Known seeds straddle the midpoint", side([0.2, 0.8], zero, one), MIDWAY),
        ("Seed spread zero everywhere", side([0.6, 0.6], zero, one), "like anchor 2"),
        ("Beyond anchor 2", side([1.5, 1.5], zero, one), "like anchor 2, beyond it"),
        ("Anchors do not separate (E94 think at matched)", side([3.7, 3.7], e(rules.PAIRS["a1"], "think"), e(rules.PAIRS["matched"], "think")),
         "no call: the anchors do not separate"),
        ("Missing seed", side([None, 0.5], zero, one), "unread"),
        ("W0 as the known arm (no call on every reading)", side([None, None], zero, one), "unread"),
        ("Do-nothing, placement (no split)", side([None, None], a1p, a2p), "unread"),
        ("Do-nothing, outside: reads as the state does", side(a2o, a1o, a2o), "like anchor 2"),
        ("Known = anchor 1, outside, gap negative", side(a1o, a1o, a2o), "like anchor 1"),
        ("Known = anchor 2, placement", side(a2p, a1p, a2p), "like anchor 2")]
    out = [{"case": c, "got": got["verdict"], "f": got.get("f"), "noise": got.get("noise"), "expected": want, "as_named": got["verdict"] == want}
           for c, got, want in cases]
    for r in out:
        print(f"{'ok ' if r['as_named'] else 'BAD'}  {r['case']}: {r['got']} (f {r['f']}, noise {r['noise']}; named {r['expected']})", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E94 Amendment 1's scoring")
    ap.add_argument("--edges", action="store_true")
    if ap.parse_args().edges:
        sys.exit(0 if all(r["as_named"] for r in edges()) else 1)
    t = score()
    print(json.dumps({"sides": {s: {k: v["verdict"] for k, v in d.items()} for s, d in t["sides"].items()}, "predictions": t["predictions"]}, indent=1), flush=True)
