"""E94's numbers, comparisons and filed predictions, from the files run.py writes (CPU, no model, no torch). Plan:
reports/plans/2026-11-05_plan_e94-d3-rows.md; rules: results/predictions.json.

  python experiments/qwen_d3_rows/rules.py           # results/table.json from whatever is written
  python experiments/qwen_d3_rows/rules.py --edges   # the plan's edge-case table through these functions, before filing

A row record (results/<state>/<rows>.json) holds, per pass, per-row sums [whole, cut, fragment, other, pooled]: "base"
(K0, K00, K01; nll_W0, nll_FT, nll_m00, nll_m01), "new" (K10, K11 with W0's V) and, on own rows, "own" (K10, K11,
nll_m11 with the state's V′). Class index 0 is whole-question answer positions, 4 the pooled sum.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from commute import qwen_comparators as qc

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WHOLE, POOLED = 0, 4
GAIN_MIN, MIN_ROWS, BOOT = 0.05, 2, 2000  # E72's gain bar (also E45's share floor); a class needs two rows; row bootstrap

# name -> (E91 arm, E91 size, own rows, anchor, size label)
STATES: dict[str, tuple[str, str, str, int, str]] = {
    "a1_s0": ("anchor1_s0", "full", "a1", 1, "full"), "a1_s1": ("anchor1_s1", "full", "a1", 1, "full"),
    "a2_s42_matched": ("anchor2_s0", "at_norm", "tofu", 2, "matched"),
    "a2_s142_matched": ("anchor2_s142", "at_norm", "tofu", 2, "matched"),
    "a2_s42_full": ("anchor2_s0", "full", "tofu", 2, "full"), "a2_s142_full": ("anchor2_s142", "full", "tofu", 2, "full"),
    "a2_s42_past": ("anchor2_s0", "past_line", "tofu", 2, "past"),
    "a2_s142_past": ("anchor2_s142", "past_line", "tofu", 2, "past")}
# name -> (kind, the state whose size on the maths rows it is matched to, own rows read besides the maths rows)
RANDOMS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "gauss_small": ("gauss", "a1_s0", ()), "alive_small": ("alive", "a1_s0", ("a1", "tofu")),
    "gauss_big": ("gauss", "a2_s42_full", ()), "alive_big": ("alive", "a2_s42_full", ("tofu",))}
PAIRS = {"a1": ("a1_s0", "a1_s1"), **{s: (f"a2_s42_{s}", f"a2_s142_{s}") for s in ("matched", "full", "past")}}
NUMBERS = ("direct", "outside", "placement", "credit", "think")
TOFU_NOTE = ("May read a pull to the training's names, not only new content: author-question fact share 0.695 to 0.411, "
             "0.379; whole-set falls of 7.4 and 11.1 points. 37 x the template's size, so the anchors are not comparable at "
             "matched size with both learned.")


def ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


# ----------------------------------------------------------------------------- ratios over rows


def col(rec: dict[str, Any], part: str, key: str, c: int) -> np.ndarray:
    return np.array([r[c] for r in rec[part][key]], dtype=np.float64)


def ratio(num: np.ndarray, den: np.ndarray, n_pos: int, n_rows: int) -> float | None:
    """sum(num) / sum(den) over rows; None (no call) under MIN_ROWS rows in the class, or a denominator under GAIN_MIN
    nats per position, or anything not finite."""
    if n_rows < MIN_ROWS or n_pos <= 0:
        return None
    N, D = float(num.sum()), float(den.sum())
    if not (ok(N) and ok(D)) or round(D / n_pos, 9) < GAIN_MIN:
        return None
    return N / D


def half_width(num: np.ndarray, den: np.ndarray, boot: int = BOOT, seed: int = 0) -> float | None:
    """Half the 2.5-97.5 percentile range of sum(num)/sum(den) over `boot` row resamples (with replacement)."""
    n = len(num)
    if n < MIN_ROWS:
        return None
    W = np.apply_along_axis(np.bincount, 1, np.random.default_rng(seed).integers(0, n, (boot, n)), minlength=n)
    D = W @ den
    r = np.sort((W @ num)[D != 0] / D[D != 0])
    return float((r[int(0.975 * len(r)) - 1] - r[int(0.025 * len(r))]) / 2) if len(r) else None


def read(num: np.ndarray, den: np.ndarray, n_pos: int, n_rows: int) -> dict[str, Any]:
    v = ratio(num, den, n_pos, n_rows)
    return {"value": v, "half_width": half_width(num, den) if v is not None else None}


def direct(rec: dict[str, Any], c: int) -> dict[str, Any]:
    """Row 1: (K00 − K10) / (K00 − K11), the masks read with W0's V (E61), over class c."""
    K00, K10, K11 = col(rec, "base", "K00", c), col(rec, "new", "K10", c), col(rec, "new", "K11", c)
    return read(K00 - K10, K00 - K11, rec["n"]["kl"][c], rec["n"]["rows"][c])


def kept(rec: dict[str, Any], c: int, part: str, key: str) -> dict[str, Any]:
    """(masked W0 NLL − the state's masked NLL) / (clean W0 NLL − FT NLL) over class c (E68's gain, zero at masked W0)."""
    num = col(rec, "base", "nll_m00", c) - col(rec, part, key, c)
    return read(num, col(rec, "base", "nll_W0", c) - col(rec, "base", "nll_FT", c), rec["n"]["nll"][c], rec["n"]["rows"][c])


def credit(rec: dict[str, Any], c: int) -> dict[str, Any]:
    """Row 4: kept under least squares' own masks minus kept by do-nothing under its own masks; raw kept beside."""
    num = col(rec, "base", "nll_m01", c) - col(rec, "own", "nll_m11", c)
    den = col(rec, "base", "nll_W0", c) - col(rec, "base", "nll_FT", c)
    return read(num, den, rec["n"]["nll"][c], rec["n"]["rows"][c]) | {
        "raw_ls": kept(rec, c, "own", "nll_m11")["value"], "raw_do_nothing": kept(rec, c, "base", "nll_m01")["value"]}


def printed(rec: dict[str, Any]) -> dict[str, Any]:
    """Unscored, on the maths rows: K00 − K0 per position (the size there), the re-gating share."""
    n = rec["n"]["kl"][POOLED]
    K0, K00, K01, K11 = (sum(r[POOLED] for r in rec[p][k]) for p, k in (("base", "K0"), ("base", "K00"), ("base", "K01"), ("new", "K11")))
    return {"size_K00_minus_K0": (K00 - K0) / n, "regating": (K00 - K01) / (K00 - K11) if K00 != K11 else None}


# ----------------------------------------------------------------------------- one state


def load(name: str) -> dict[str, Any] | None:
    d = RESULTS / name
    if not (d / "state.json").is_file():
        return None
    return {"state": json.loads((d / "state.json").read_text()),
            "rows": {p.stem: json.loads(p.read_text()) for p in d.glob("*.json") if p.stem != "state"}}


def numbers(rec: dict[str, Any], own: str | None) -> dict[str, Any]:
    """The five numbers of one state (None: no call), with what is printed beside them."""
    st, rows = rec["state"], rec["rows"]
    pl, th = st.get("placement") or {}, st.get("overlap") or {}
    T, L = ((th.get(k) or {}).get("fold") for k in ("T", "L"))
    out: dict[str, Any] = {
        "direct_maths": direct(rows["maths"], POOLED), "outside": {"value": st.get("outside"), "null": st.get("outside_null")},
        "placement": {"value": pl.get("excess"), "share": pl.get("share"), "null": pl.get("null")},
        "think": {"value": T, "L": L, "T_minus_L": T - L if T is not None and L is not None else None, "size_S": th.get("size_S")},
        "printed": printed(rows["maths"])}
    if own is not None and own in rows:
        out["direct"] = direct(rows[own], WHOLE) | {"pooled": direct(rows[own], POOLED)["value"]}
        if "own" in rows[own]:
            c = credit(rows[own], WHOLE)
            out["credit"] = c | {"pooled": credit(rows[own], POOLED)["value"]}
    return out


def value(nums: dict[str, Any] | None, number: str) -> float | None:
    if nums is None:
        return None
    v = (nums.get(number) or {}).get("value")
    return v if ok(v) else None


# ----------------------------------------------------------------------------- comparisons


def compare(nums: dict[str, Any], number: str, size: str) -> dict[str, Any]:
    """`qc.separates` between anchor 1's two seeds and anchor 2's at `size`, each value's half-width beside."""
    a1 = [value(nums.get(n), number) for n in PAIRS["a1"]]
    a2 = [value(nums.get(n), number) for n in PAIRS[size]]
    hw = {n: ((nums.get(n) or {}).get(number) or {}).get("half_width") for n in (*PAIRS["a1"], *PAIRS[size])}
    return {"anchor1": a1, "anchor2": a2, "verdict": qc.separates(a1, a2), "half_widths": hw}


def size_can(nums: dict[str, Any], number: str) -> str:
    """The same rule between the two random sizes, the two kinds as the pair at each size."""
    get = lambda n: value(nums.get(n), number)
    return qc.separates([get("gauss_small"), get("alive_small")], [get("gauss_big"), get("alive_big")])


def comparisons(nums: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for size in ("matched", "full", "past"):
        out[size] = {k: compare(nums, k, size) for k in (*NUMBERS, "direct_maths")}
        d = out[size]
        d["direct"]["rows"] = ("depends on the rows" if {d["direct"]["verdict"], d["direct_maths"]["verdict"]} == {"counts", "does not count"}
                               else "same on both row sets")
    for k in ("direct_maths", "outside", "placement", "think"):
        out["full"][k]["size_can_do_this"] = size_can(nums, k)
    counted = [f"{k} at {s}" for s in ("matched", "full") for k in NUMBERS if out[s][k]["verdict"] == "counts"]
    out["stop_rule"] = {"counted": counted, "reads": "some number counts" if counted else
                        "no number counts at the full update or the matched step: that is D3's finding"}
    return out


# ----------------------------------------------------------------------------- the filed predictions


def per_seed(xs: list[float | None], test: Any) -> str:
    if any(x is None for x in xs):
        return "unread"
    return "holds" if all(test(x) for x in xs) else "misses"


def predictions(nums: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    t = P["thresholds"]
    g = lambda n, k="direct": value(nums.get(n), k)
    rnd = lambda n, rows: value((nums.get(n) or {}).get("own_rows", {}).get(rows), "direct")
    a1r, a2s, a2b = rnd("alive_small", "a1"), rnd("alive_small", "tofu"), rnd("alive_big", "tofu")
    out = {
        "1_a1": per_seed([g(n) for n in PAIRS["a1"]], lambda x: a1r is not None and abs(x - a1r) <= t["p1_within"]) if a1r is not None else "unread",
        "1_a2": per_seed([g(n) for n in PAIRS["matched"]], lambda x: x < a2s) if a2s is not None else "unread",
        "2": per_seed([g(n) for n in PAIRS["full"]], lambda x: x < a2b) if a2b is not None else "unread"}
    p3 = []
    for pair in (PAIRS["a1"], PAIRS["matched"], PAIRS["full"], PAIRS["past"]):
        v = [g(n, "outside") for n in pair]
        nulls = [((nums.get(n) or {}).get("outside") or {}).get("null") for n in pair]
        if any(x is None for x in v) or not all(ok(x) for x in nulls):
            p3.append("unread")
            continue
        spread = abs(v[0] - v[1])  # type: ignore[operator]
        p3.append("holds" if all(nl - x > spread for x, nl in zip(v, nulls, strict=True)) else "misses")  # type: ignore[operator]
    out["3"] = "unread" if "unread" in p3 else "holds" if set(p3) == {"holds"} else "misses"
    p4 = [compare(nums, "placement", s)["verdict"] for s in ("matched", "full")]
    out["4"] = "unread" if "unread" in p4 else "holds" if set(p4) == {"does not count"} else "misses"
    out["5"] = per_seed([g(n, "credit") for n in PAIRS["a1"]], lambda x: x < t["p5_below"])
    states = [n for k in ("a1", "matched", "full", "past") for n in PAIRS[k]]
    tl = [((nums.get(n) or {}).get("think") or {}).get("T_minus_L") for n in states]
    out["6_T_like_L"] = per_seed(tl, lambda x: abs(x) <= t["p6_t_minus_l"])
    every = [((nums.get(n) or {}).get("think") or {}).get(k) for n in (*states, *RANDOMS) for k in ("value", "L")]
    out["6_fold_3_to_4"] = per_seed(every, lambda x: t["p6_fold"][0] <= x <= t["p6_fold"][1])
    return out


def score() -> dict[str, Any]:
    nums: dict[str, Any] = {}
    for n, (_, _, own, _, _) in STATES.items():
        rec = load(n)
        if rec is not None:
            nums[n] = numbers(rec, own) | ({"note": TOFU_NOTE} if n.startswith("a2") and not n.endswith("matched") else {})
    for n, (_, _, rows) in RANDOMS.items():
        rec = load(n)
        if rec is not None:
            nums[n] = numbers(rec, None) | {"own_rows": {r: numbers(rec, r) for r in rows if r in rec["rows"]}}
    P = json.loads((RESULTS / "predictions.json").read_text())
    table = {"numbers": nums, "comparisons": comparisons(nums), "predictions": predictions(nums, P)}
    tmp = RESULTS / "table.json.tmp"
    tmp.write_text(json.dumps(table, indent=1, default=float) + "\n")
    tmp.replace(RESULTS / "table.json")
    return table


# ----------------------------------------------------------------------------- the plan's edge cases


def fake(K00: float, K10: float, K11: float, rows: int = 4, n: int = 256) -> dict[str, Any]:
    """A row record whose every row reads these per-position KLs, whole-question class only."""
    r = lambda x: [[x * n, 0.0, 0.0, 0.0, x * n]] * rows
    return {"n": {"kl": [n * rows, 0, 0, 0, n * rows], "nll": [n * rows, 0, 0, 0, n * rows], "rows": [rows, 0, 0, 0, rows]},
            "base": {"K00": r(K00), "nll_m00": r(K00), "nll_m01": r(K11), "nll_W0": r(K00), "nll_FT": r(0.0)},
            "new": {"K10": r(K10), "K11": r(K11)}, "own": {"nll_m11": r(K11)}}


def edges() -> list[dict[str, Any]]:
    """Each case of the plan's table, the outcome it names, and what these functions give."""
    dn = direct(fake(1.0, 1.0, 0.1), WHOLE)["value"]  # do-nothing: no column moves, K10 = K00
    w0 = direct(fake(0.07, 0.07, 0.07), WHOLE)["value"]  # W0: K00 = K10 = K11 = K0
    small = direct(fake(0.06, 0.05, 0.02), WHOLE)["value"]  # a denominator of 0.04 nats per position
    one_row = direct(fake(1.0, 0.9, 0.1, rows=1), WHOLE)["value"]
    raw_dn = kept(fake(1.0, 1.0, 0.3), WHOLE, "base", "nll_m01")["value"]  # do-nothing's raw kept, 0.7 here
    cases = [
        ("Gap equal to spread", qc.separates([0.0, 1.0], [1.5, 1.5]), "does not count"),
        ("Spread exactly 0", qc.separates([0.1, 0.1], [0.1001, 0.1001]), "counts"),
        ("Do-nothing on both anchors, row 1", (dn, qc.separates([dn, dn], [dn, dn])), (0.0, "does not count")),
        ("Do-nothing, raw kept (could pass: so it is not scored)", qc.separates([raw_dn, raw_dn], [0.2, 0.3]), "counts"),
        ("Do-nothing, credit", credit(fake(1.0, 1.0, 0.3), WHOLE)["value"], 0.0),
        ("W0 as both anchors", (w0, qc.separates([w0, w0], [w0, w0])), (None, "unread")),
        ("Zero denominator, gain under 0.05", (small, qc.separates([small, 0.1], [0.2, 0.3])), (None, "unread")),
        ("Empty set: no fold", qc.separates([None, 3.4], [3.5, 3.6]), "unread"),
        ("Under 2 rows in a class", one_row, None),
        ("Missing seed", qc.separates([0.1], [0.2, 0.3]), "unread"),
        ("Matched-step gap", qc.separates([0.01, 0.012], [0.03, 0.031]), "counts")]
    out = [{"case": c, "got": got, "expected": want, "as_named": got == want} for c, got, want in cases]
    for r in out:
        print(f"{'ok ' if r['as_named'] else 'BAD'}  {r['case']}: {r['got']} (named {r['expected']})", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E94's scoring")
    ap.add_argument("--edges", action="store_true")
    if ap.parse_args().edges:
        sys.exit(0 if all(r["as_named"] for r in edges()) else 1)
    t = score()
    print(json.dumps({"comparisons": t["comparisons"]["stop_rule"], "predictions": t["predictions"]}, indent=1), flush=True)
