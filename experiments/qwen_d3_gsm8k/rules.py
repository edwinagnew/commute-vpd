"""E104's numbers, comparisons and draft predictions, from the files run.py writes and E94's state files (CPU, no model,
no torch). Plan: reports/plans/2026-11-06_plan_e104-d3-gsm8k.md ("The counting rule"); rules: results/predictions.json.

  python experiments/qwen_d3_gsm8k/rules.py           # results/table.json from whatever is written
  python experiments/qwen_d3_gsm8k/rules.py --edges   # the plan's edge-case table on synthetic inputs (results/edges.txt)

A state record (results/<state>/state.json, keyed as E94's) holds "outside", "outside_null", "placement" {share, null,
excess}, "overlap" {size_S, T, L: `qr.overlap` with fold and draw_sd} and "size_K00_minus_K0"; a scaled copy adds
"search" (`qr.scale_search`). Every number reads None (no call) when not finite.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from commute import qwen_comparators as qc
from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E94 = EXPERIMENTS / "qwen_d3_rows" / "results"
STEPS = (10, 100)
ARMS = {"R": ("R_s0", "R_s70"), "N": ("N_s0", "N_s70"), "F": ("F_s0",)}  # RL, shuffled-reward null, own-samples SFT
NULL_ARM = "N"
STATES = {f"{a}_{s}": (a, s) for seeds in ARMS.values() for a in seeds for s in STEPS}  # name -> (E91 arm, step)
E94_STATES = {"a1_s0": ("anchor1_s0", "full"), "a1_s1": ("anchor1_s1", "full"), "a2_s42_full": ("anchor2_s0", "full"),
              "a2_s142_full": ("anchor2_s142", "full"), "a2_s42_matched": ("anchor2_s0", "at_norm"),
              "a2_s142_matched": ("anchor2_s142", "at_norm")}  # name -> (E91 arm, size): E94's anchors, read from E94
ANCHORS = {"template": ("a1_s0", "a1_s1"), "tofu_full": ("a2_s42_full", "a2_s142_full"),
           "tofu_matched": ("a2_s42_matched", "a2_s142_matched")}
# scaled copy -> (source state, the state whose size it is matched to); an "a.." name is E94's
COPIES = {"S1": ("R_s0_100", "F_s0_100"), "S2": ("R_s0_100", "R_s0_10"), "S3": ("a2_s42_full", "a1_s0")}
NUMBERS = ("outside", "placement", "think")
CHANCE_MIN = 5.0  # Cochran 1954: an expected count of at least 5 before a chance comparison is read


def ok(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def fin(x: Any) -> float | None:
    return float(x) if ok(x) else None


def frac(x: float | None, y: float | None) -> float | None:
    return x / y if x is not None and y is not None and y != 0 else None


# ----------------------------------------------------------------------------- one state


def think(ov: dict[str, Any] | None, ref: str) -> float | None:
    """The fold |S ∩ ref| / its chance mean; no call for an empty S or a chance mean under CHANCE_MIN."""
    x = (ov or {}).get(ref) or {}
    if not (ov or {}).get("size_S") or not ok(x.get("mean")) or x["mean"] < CHANCE_MIN:
        return None
    return fin(x["o"] / x["mean"])


def numbers(st: dict[str, Any]) -> dict[str, Any]:
    """The three numbers of one state record (value None: no call), with what is printed beside them."""
    pl, ov = st.get("placement") or {}, st.get("overlap") or {}
    nulls = [fin(x) for x in pl.get("null") or []]
    pnull = sum(nulls) / len(nulls) if nulls and None not in nulls else None  # type: ignore[arg-type]
    out_v, out_n, share = fin(st.get("outside")), fin(st.get("outside_null")), fin(pl.get("share"))
    T = ov.get("T") or {}
    return {"outside": {"value": out_v, "null": out_n, "fraction_of_null": frac(out_v, out_n)},
            "placement": {"value": fin(pl.get("excess")) if share is not None and pnull is not None else None,
                          "share_on_set": share, "null": pnull, "fraction_of_null": frac(share, pnull)},
            "think": {"value": think(ov, "T"), "L": think(ov, "L"), "size_S": ov.get("size_S"),
                      "chance_mean": T.get("mean"), "draw_sd": T.get("draw_sd")},
            "size_K00_minus_K0": fin(st.get("size_K00_minus_K0")),
            "matched": (st.get("search") or {}).get("matched", True)}


def load(name: str) -> dict[str, Any] | None:
    p = (E94 if name in E94_STATES else RESULTS) / name / "state.json"
    return json.loads(p.read_text()) if p.is_file() else None


def value(nums: dict[str, Any], name: str, number: str) -> float | None:
    return ((nums.get(name) or {}).get(number) or {}).get("value")


# ----------------------------------------------------------------------------- the counting rule


def mean(xs: list[float | None]) -> float | None:
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)  # type: ignore[arg-type]


def spread(xs: list[float | None]) -> float | None:
    """|seed a − seed b|; None unless both seeds read."""
    return abs(xs[0] - xs[1]) if len(xs) == 2 and None not in xs else None  # type: ignore[operator]


def exceeds(gap: float | None, spreads: list[float | None]) -> str:
    """`qc.separates`'s test for a gap formed some other way: "counts" when |gap| is strictly over the largest spread."""
    if gap is None or not spreads or any(s is None for s in spreads):
        return "unread"
    return "counts" if abs(gap) > max(spreads) else "does not count"  # type: ignore[type-var]


def vals(nums: dict[str, Any], arm: str, step: int, number: str) -> list[float | None]:
    return [value(nums, f"{a}_{step}", number) for a in ARMS[arm]]


def bar(nums: dict[str, Any], arm: str, step: int, number: str) -> float | None:
    """An arm's seed spread; a one-seed arm borrows the null arm's at the same step and number (E93's rule)."""
    return spread(vals(nums, arm if len(ARMS[arm]) == 2 else NULL_ARM, step, number))


def c1(nums: dict[str, Any], step: int, number: str) -> str:
    return qc.separates(vals(nums, "R", step, number), vals(nums, "N", step, number))


def c2(nums: dict[str, Any], step: int, number: str) -> str:
    F, R = vals(nums, "F", step, number)[0], vals(nums, "R", step, number)
    m = mean(R)
    return exceeds(F - m if F is not None and m is not None else None, [spread(R), bar(nums, "F", step, number)])


def c3(nums: dict[str, Any], arm: str, number: str) -> str:
    a, b = (vals(nums, arm, s, number) for s in STEPS)
    if len(ARMS[arm]) == 2:
        return qc.separates(a, b)
    return exceeds(a[0] - b[0] if a[0] is not None and b[0] is not None else None, [bar(nums, arm, s, number) for s in STEPS])


def size_check(nums: dict[str, Any], copy: str, small: list[float | None], spreads: list[float | None], verdict: str) -> dict[str, Any]:
    """The scaled copy in place of the larger state against the smaller side, the original comparison's bar; think only."""
    cp, m = nums.get(copy), mean(small)
    if cp is None or not cp["matched"]:
        return {"verdict": verdict, "with_copy": "unread", "reads": "size unmatched" if cp else "copy not read"}
    v = value(nums, copy, "think")
    w = exceeds(v - m if v is not None and m is not None else None, spreads)
    reads = {"counts": "size does not explain it", "does not count": "size can do this"}.get(w, "unread")
    return {"verdict": verdict, "copy": v, "smaller_mean": m, "with_copy": w, "reads": reads}


def sizes(nums: dict[str, Any]) -> dict[str, Any]:
    th = lambda names: [value(nums, n, "think") for n in names]
    sp = lambda arm, s: spread(vals(nums, arm, s, "think"))
    a1, a2 = th(ANCHORS["template"]), th(ANCHORS["tofu_full"])
    return {"S1_C2_at_100": size_check(nums, "S1", [value(nums, "F_s0_100", "think")], [sp("R", 100), sp("N", 100)], c2(nums, 100, "think")),
            "S2_C3_RL": size_check(nums, "S2", vals(nums, "R", 10, "think"), [sp("R", 10), sp("R", 100)], c3(nums, "R", "think")),
            "S3_anchors_full": size_check(nums, "S3", a1, [spread(a1), spread(a2)], qc.separates(a1, a2))}


def position(nums: dict[str, Any], arm: str, number: str, against: str = "tofu_full") -> dict[str, Any]:
    """p = (arm mean − template mean) / (`against` mean − template mean) at step 100; no call on a zero denominator."""
    m = mean(vals(nums, arm, 100, number))
    t, o = (mean([value(nums, n, number) for n in ANCHORS[k]]) for k in ("template", against))
    p = frac(m - t, o - t) if m is not None and t is not None and o is not None else None
    label = None if p is None else "beyond" if not 0 <= p <= 1 else "nearer the template" if p < 0.5 else "nearer TOFU"
    return {"p": p, "reads": label}


def comparisons(nums: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: {} for k in ("C1", "C2", "C3", "position")}
    for k in NUMBERS:
        out["C1"][k] = {str(s): c1(nums, s, k) for s in STEPS}
        out["C2"][k] = {str(s): c2(nums, s, k) for s in STEPS}
        out["C3"][k] = {a: c3(nums, a, k) for a in ARMS}
        out["position"][k] = {a: position(nums, a, k) | {"p_vs_tofu_matched": position(nums, a, k, "tofu_matched")["p"]} for a in ARMS}
    out["size_think"] = sizes(nums)
    counted = [k for k in NUMBERS if out["C1"][k]["100"] == "counts"]
    out["C1_at_100"] = {"counted": counted, "reads": "the reward's content moves " + ", ".join(counted) if counted else
                        "RL reads like its null on every carried row" if all(out["C1"][k]["100"] != "unread" for k in NUMBERS) else "unread"}
    out["C2_branch"] = "fires: ask for own-samples seed 70" if any(out["C2"][k]["100"] == "counts" for k in NUMBERS) else "does not fire"
    return out


# ----------------------------------------------------------------------------- the draft predictions


def every(tests: list[bool | None]) -> str:
    if not tests or any(t is None for t in tests):
        return "unread"
    return "holds" if all(tests) else "misses"


def predictions(nums: dict[str, Any], P: dict[str, Any]) -> dict[str, str]:
    t = P["thresholds"]
    p1: list[bool | None] = []
    for arm, seeds in ARMS.items():
        for s in STEPS:
            b = bar(nums, arm, s, "outside")
            for a in seeds:
                o = (nums.get(f"{a}_{s}") or {}).get("outside") or {}
                v, n = o.get("value"), o.get("null")
                p1.append(None if v is None or n is None or b is None else n - v > b)
    p2: list[bool | None] = []
    for a in (a for seeds in ARMS.values() for a in seeds):
        x, y = (value(nums, f"{a}_{s}", "placement") for s in STEPS)
        p2.append(None if x is None or y is None else x - y > t["p2_margin"])
    s3 = nums.get("S3")
    v3 = value(nums, "S3", "think")
    p3 = "unread" if s3 is None or not s3["matched"] or v3 is None else "holds" if v3 < t["p3_bar"] else "misses"
    v4 = c1(nums, 100, "outside")
    return {"1": every(p1), "2": every(p2), "3": p3, "4": "unread" if v4 == "unread" else "holds" if v4 == "does not count" else "misses"}


def table(nums: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    return {"numbers": nums, "comparisons": comparisons(nums), "predictions": predictions(nums, P),
            "note": "Outside share and think overlap describe the update: do-nothing reads exactly as each state, so these "
                    "two rows can pass without any split. Placement under do-nothing is no call."}


def score() -> dict[str, Any]:
    nums = {n: numbers(st) for n in (*STATES, *COPIES, *E94_STATES) if (st := load(n)) is not None}
    t = table(nums, json.loads((RESULTS / "predictions.json").read_text()))
    tmp = RESULTS / f"table.json.{os.getpid()}.tmp"  # two jobs may score at once
    tmp.write_text(json.dumps(t, indent=1, default=float) + "\n")
    tmp.replace(RESULTS / "table.json")
    return t


def invariant(copy: dict[str, Any], source: dict[str, Any], band: float) -> tuple[bool, float]:
    """A scaled copy's outside share, its null and placement excess against its source's: the largest absolute gap,
    and whether it is within `band` (a non-finite reading fails)."""
    gaps = [abs(copy[k]["value"] - source[k]["value"]) if copy[k]["value"] is not None and source[k]["value"] is not None
            else math.inf for k in ("outside", "placement")]
    a, b = copy["outside"]["null"], source["outside"]["null"]
    gaps.append(abs(a - b) if a is not None and b is not None else math.inf)
    return max(gaps) <= band, max(gaps)


# ----------------------------------------------------------------------------- the plan's edge cases


def fake(outside: float | None, null: float | None, excess: float | None, o: int, chance: float, size_S: int = 100,
         matched: bool = True) -> dict[str, Any]:
    """A state record reading these values (think: o hits against a chance mean)."""
    ov = {"size_S": size_S, **{k: {"o": o, "mean": chance, "sd": 1.0, "draw_sd": 1.0 / chance if chance else None} for k in ("T", "L")}}
    pl = None if excess is None else {"share": 0.5 + excess, "null": [0.5] * 4, "excess": excess}
    st = {"outside": outside, "outside_null": null, "placement": pl, "overlap": ov, "size_K00_minus_K0": 0.1}
    return numbers(st | ({"search": {"matched": matched}} if not matched else {}))


def world(dn: bool = False) -> dict[str, Any]:
    """Every E104 state and E94 anchor with distinct values, binary fractions so that ties are exact: outside under
    its null by far, placement falling 1/32 from step 10 to 100. `dn`: the do-nothing reading of each, the same outside
    and think (they describe δW), placement no call."""
    return {n: fake(0.125 + i / 1024, 0.5, None if dn else 1 / 32 if n.endswith("_100") or n.startswith("a") else 1 / 16,
                    300 + i, 64.0) for i, n in enumerate((*STATES, *E94_STATES))}


def outside_world(F: float) -> dict[str, Any]:
    """Outside share alone, binary: RL and null seeds 0.125 and 0.25 (spread 0.125), own-samples F, null 0.5."""
    v = {"R_s0": 0.125, "R_s70": 0.25, "N_s0": 0.125, "N_s70": 0.25, "F_s0": F}
    return {n: fake(v[a], 0.5, 1 / 16 if s == 10 else 1 / 32, 300, 64.0) for n, (a, s) in STATES.items()}


def edges() -> list[tuple[str, Any, Any]]:
    w, dn = world(), world(dn=True)
    one = {**w, "R_s0_100": fake(None, 0.5, 0.0, 300, 64.0)}
    no_null = {**w, "N_s70_100": fake(0.125, 0.5, None, 0, 0.0, size_S=0)}
    small_chance = fake(0.125, 0.5, 0.0, 30, 4.99)
    zero = fake(math.nan, math.nan, math.nan, 0, 0.0, size_S=0)
    flat = {**w, **{n: fake(0.125, 0.5, 0.0, 300, 64.0) for n in E94_STATES}}
    P = {"thresholds": {"p2_margin": 1 / 128, "p3_bar": 3.5}}
    p2_tie = {**w, "R_s0_10": fake(0.125, 0.5, 1 / 64, 300, 64.0), "R_s0_100": fake(0.125, 0.5, 1 / 128, 300, 64.0)}
    s3 = lambda o: {**w, "S3": fake(0.125, 0.5, 0.0, o, 1000.0)}
    eq = {**w, **{f"N_{a}_{s}": w[f"R_{a}_{s}"] for a in ("s0", "s70") for s in STEPS}}
    S1 = {**w, "S1": fake(0.125, 0.5, 0.0, 300, 64.0, matched=False)}
    good = fake(0.1, 0.2, 0.01, 300, 64.0)
    dn_verdicts = lambda x: [f(x, k) for k in ("outside", "think") for f in (lambda y, k: c1(y, 100, k), lambda y, k: c2(y, 100, k))]
    return [
        ("Gap equal to the bar (separates)", qc.separates([0.0, 1.0], [1.5, 1.5]), "does not count"),
        ("Gap equal to the bar (borrowed)", exceeds(0.5, [0.25, 0.5]), "does not count"),
        ("Both spreads exactly 0, non-zero gap", qc.separates([0.1, 0.1], [0.1001, 0.1001]), "counts"),
        ("Both spreads exactly 0, borrowed", exceeds(1e-9, [0.0, 0.0]), "counts"),
        ("One seed: F borrows the null's spread", (c2(w, 10, "outside"), c3(w, "F", "outside")), ("counts", "does not count")),
        ("A seed no call (C1, C2, C3 RL)", (c1(one, 100, "outside"), c2(one, 100, "outside"), c3(one, "R", "outside")), ("unread",) * 3),
        ("Null spread unread for a borrowed bar", (c2(no_null, 100, "think"), c3(no_null, "F", "think")), ("unread", "unread")),
        ("Think, chance mean under 5", small_chance["think"]["value"], None),
        ("Think, empty S", fake(0.1, 0.2, 0.0, 0, 0.0, size_S=0)["think"]["value"], None),
        ("Think no call makes the comparison unread", c1({**w, "R_s0_10": small_chance}, 10, "think"), "unread"),
        ("Outside, placement zero denominator (NaN)", (zero["outside"]["value"], zero["placement"]["value"]), (None, None)),
        ("p with TOFU full mean = template mean", position(flat, "R", "outside")["p"], None),
        ("Do-nothing, outside and think: C1 and C2 as the states'", dn_verdicts(dn), dn_verdicts(w)),
        ("Do-nothing, placement: no call, cannot pass", (c1(dn, 100, "placement"), predictions(dn, P)["2"]), ("unread", "unread")),
        ("Do-nothing passes prediction 1 as the states do", (predictions(dn, P)["1"], predictions(w, P)["1"]), ("holds", "holds")),
        ("Scaled copy unmatched: size check unread", sizes(S1)["S1_C2_at_100"]["with_copy"], "unread"),
        ("Scaled copy off its source by over 1e-4: stop", invariant(fake(0.1002, 0.2, 0.01, 300, 64.0), good, 1e-4)[0], False),
        ("Scaled copy within 1e-4 of its source", invariant(fake(0.10005, 0.2, 0.01, 300, 64.0), good, 1e-4)[0], True),
        ("Scaled copy with a no-call reading: stop", invariant(zero, good, 1e-4)[0], False),
        ("RL seeds equal to null seeds", c1(eq, 100, "outside"), "does not count"),
        ("Prediction 1, null - value equal to the borrowed bar (strict); just over", (predictions(outside_world(0.375), P)["1"],
         predictions(outside_world(0.25), P)["1"]), ("misses", "holds")),
        ("Prediction 2, fall equal to the margin (strict); falls of 1/32", (predictions(p2_tie, P)["2"], predictions(w, P)["2"]),
         ("misses", "holds")),
        ("Prediction 3, S3 at the bar (strict); under; unread before S3", (predictions(s3(3500), P)["3"],
         predictions(s3(3400), P)["3"], predictions(w, P)["3"]), ("misses", "holds", "unread")),
        ("Prediction 4 unread with a seed missing", predictions(one, P)["4"], "unread"),
    ]


def run_edges() -> bool:
    lines = [f"{'ok ' if got == want else 'BAD'}  {case}: {got} (named {want})" for case, got, want in edges()]
    (RESULTS / "edges.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return all(line.startswith("ok") for line in lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E104's scoring")
    ap.add_argument("--edges", action="store_true")
    if ap.parse_args().edges:
        sys.exit(0 if run_edges() else 1)
    t = score()
    print(json.dumps({"comparisons": {k: t["comparisons"][k] for k in ("C1", "C2", "C3", "size_think", "C1_at_100")},
                      "predictions": t["predictions"]}, indent=1, default=float), flush=True)
