"""E111's stages, rates, early checks, verdicts and filed leans, from the files run.py writes (CPU, no torch, no model).
Plan: reports/plans/2026-11-06_plan_e111-teacher-table.md, with its "Overrides (user, 2026-11-06)": a state passes or
fails on the behaviour columns alone (1 clean, 4 kept clean, 5 K under own masks, 6 kept under own masks), each read by
E93's rule functions unchanged (experiments/qwen_d1_table/rules.py); columns 2 (share left outside the components) and
3 (squared size) are printed beside, not judged. Δ is frozen: the masked cells' Γ reads W0 + moved part.

  python experiments/qwen_teacher_table/rules.py           # results/table.json from the states written so far
  python experiments/qwen_teacher_table/rules.py --edges   # each rule and lean on its edge cases, before filing

results/<state>/state.json: "shared" (this card's clean W0, FT and least squares' moved part; the split's sums; the
base cells; least squares' own cells at FT's weights, E93's path, and with Δ frozen) and "arms" (per arm: training
readings, the kept state's split sums, clean moved part and remainder, gain, own cells). Reads that do not depend on
the map and are not re-read here (random, signs, the gain rows' W0, FT and t, the ceiling, the random under masks) are
E93's (`E93.load`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment

E93: Any = load_experiment("qwen_d1_table", "rules")
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HOLDS, MISSES, FAILS, NO_CALL, PENDING = E93.HOLDS, E93.MISSES, E93.FAILS, E93.NO_CALL, E93.PENDING
STAGES: dict[int, tuple[str, ...]] = {1: ("R_s70",), 2: ("N_s0", "N_s70", "F_s0"), 3: ("a1_s0", "a1_s1", "R_s0"),
                                      4: ("a2_s42", "a2_s142"), 5: ("u1", "u2")}
ARMS: dict[int, tuple[tuple[str, float], ...]] = {1: (("1x", 1.0), ("10x", 10.0))} | {k: (("1x", 1.0),) for k in (2, 3, 4, 5)}
# lr = 1e-6 × √(room / 0.72) per arm label (plan appendix C); the no-room arms take RL s70's (appendix B)
RATE = {"anchor 1": 3.0e-8, "anchor 2": 4.6e-7, "RL": 1.3e-8, "null": 1.3e-8, "own-samples": 1.3e-8, "u1": 1.0e-7, "u2": 7.8e-8}
WALK_PER_STEP = 1.2e-3  # ‖r‖² per step at lr 1e-6, E76's arm A from 0.52 at step 100 to 1.81 at 1200 (appendix C)
STEPS, EVERY = 300, 25  # E76's control length (common.py:63); readings every 25 (plan section 2)
THRESHOLDS = {"excess_ratio": [30.0, 300.0]}
FIRST = {"norm_rel": 1e-3, "ft_zero": 1e-6, "card_max": 1e-3, "band_floor": 1e-5, "band_factor": 10.0, "split_rel": 1e-5,
         "tf32_rel": 1e-3, "walk": [0.3, 3.0], "sqrt_v_min": 1e-7, "gpu_max_gib": 76.0, "consistency_rel": 1e-5}
BEHAVIOUR_COLUMNS = ("1", "4", "5", "6")


def forecast(label: str, scale: float, step: int) -> float:
    return WALK_PER_STEP * (RATE[label] * scale / 1e-6) ** 2 * step


def rel(x: float, ref: float) -> float:
    return abs(x / ref - 1) if ref != 0 else math.inf


# ----------------------------------------------------------------------------- early checks (each a stop on a miss)


def plateau(ks: list[float]) -> bool:
    """Training's end before `STEPS`: the last four 8-row K readings all at or above the lowest reading before them
    (the plan's "has not fallen over four readings"); five readings at least, step 0 included."""
    return len(ks) >= 5 and min(ks[-4:]) >= min(ks[:-4])


def band(drift: float) -> float:
    """The reproduction band for this card: the floor, or band_factor × the measured drift of a map-free read from E93's."""
    return max(FIRST["band_floor"], FIRST["band_factor"] * drift)


def judge(name: str, v: dict[str, Any]) -> bool:
    """Whether an early check holds; `v` holds the reading and its reference. A NaN anywhere misses."""
    F = FIRST
    if any(isinstance(x, float) and not math.isfinite(x) for x in v.values()):
        return False
    if name == "norm":
        return rel(v["norm"], v["e93"]) <= F["norm_rel"] and v["sha256"] == v["e93_sha256"]
    if name == "card":  # map-free reads against E93's: the ruler (clean W0) or W0 masked (K00)
        return rel(v["here"], v["e93"]) <= F["card_max"]
    if name == "ft_zero":
        return v["K"] <= F["ft_zero"]
    if name == "split":
        return all(rel(v[k], v[f"e93_{k}"]) <= F["split_rel"] for k in ("share", "size"))
    if name == "reproduce":  # a least-squares read against E93's stored one, inside this card's band
        return all(rel(v[k], v[f"e93_{k}"]) <= v["band"] for k in v["keys"])
    if name == "zero_offsets":
        return v["r2"] == 0 and rel(v["tf32_K8"], v["K8"]) <= F["tf32_rel"] and rel(v["K8"], v["e93_K8"]) <= v["band"]
    if name == "gradient":
        return v["dV"] > 0 and v["dU"] > 0
    if name == "first_reading":
        lo, hi = F["walk"]
        return (lo * v["forecast_r2"] <= v["r2"] <= hi * v["forecast_r2"] and v["median_sqrt_v"] > F["sqrt_v_min"]
                and E93.r9(v["K8"]) <= E93.r9(v["K8_step0"]) and v["gpu_gib"] <= F["gpu_max_gib"])
    if name == "consistency":
        return v["gap"] <= F["consistency_rel"]
    raise KeyError(name)


# ----------------------------------------------------------------------------- one state's records and verdicts


def table_record(e93: dict[str, Any], shared: dict[str, Any], arm: dict[str, Any] | None) -> dict[str, Any]:
    """The record E93's rules read, for least squares with Δ frozen (arm None) or for a trained arm."""
    keep = ("name", "arm", "rows", "n", "ceiling", "random")
    rec = {k: e93[k] for k in keep if k in e93} | {"done": True, "base": shared["base"]}
    rec["clean"] = {k: e93["clean"][k] for k in ("random", "signs")} | {k: shared["clean"][k] for k in ("W0", "FT")}
    rec["fractions"] = dict(e93["fractions"])
    if arm is None:
        rec["clean"] |= {"moved": shared["clean"]["ls_moved"], "remainder": e93["clean"]["remainder"]}
        return rec | {"split": shared["split"], "own": shared["own_frozen"]}
    rec["clean"] |= {"moved": arm["clean"]["moved"], "remainder": arm["clean"]["remainder"]}
    rec["fractions"]["moved"] = arm["gain"]
    return rec | {"split": arm["split"], "own": arm["own"]}


def verdict(r: dict[str, str]) -> str:
    """Behaviour columns only (user, 2026-11-06): fails on a miss of 1 or 5 or "fails" on 4 or 6; passes when 1 holds or
    is at floor and 5 holds; pending while one is unread; otherwise no call."""
    if MISSES in (r["1"], r["5"]) or FAILS in (r["4"], r["6"]):
        return FAILS
    if PENDING in [r[k] for k in BEHAVIOUR_COLUMNS]:
        return PENDING
    return "passes" if r["1"] in (HOLDS, "at floor") and r["5"] == HOLDS else NO_CALL


def scored(rec: dict[str, Any], spread: float | None) -> dict[str, Any]:
    s = E93.score_state(rec, spread)
    sp = rec["split"]
    s["verdict"] = verdict({k: v["result"] for k, v in s["rules"].items()})
    s["printed"] |= {"share": sp["left"] / sp["size"], "squared_size": sp["moved"] / sp["size"],
                     "share_null": sp["null"] / sp["size"]}
    s.pop("note", None)
    return s


def score_state(e93: dict[str, Any], rec: dict[str, Any], spreads: dict[str, Any]) -> dict[str, Any]:
    sp = spreads[e93["arm"]][0]
    out: dict[str, Any] = {"arm": e93["arm"], "ls_frozen": scored(table_record(e93, rec["shared"], None), sp), "arms": {}}
    for a, r in rec.get("arms", {}).items():
        if r.get("done"):
            t = scored(table_record(e93, rec["shared"], r), sp)
            t["printed"] |= {"r2": r["train"]["readings"][-1]["r2"], "kept_step": r["train"]["kept_step"],
                             "excess_share": t["printed"]["share"] - out["ls_frozen"]["printed"]["share"]}
            out["arms"][a] = t
    return out


# ----------------------------------------------------------------------------- the filed leans


def lean(states: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    """Each filed lean's result: "unread" until its states are written; otherwise holds / misses / no call."""
    out: dict[str, Any] = {}
    one = [n for ns in STAGES.values() for n in ns]

    def arm(n: str, a: str) -> dict[str, Any] | None:
        return states.get(n, {}).get("arms", {}).get(a)

    def r1(n: str, a: str) -> str:
        x = arm(n, a)
        return "unread" if x is None else HOLDS if x["rules"]["1"]["result"] in (HOLDS, "at floor") else MISSES

    out["T1_rl70_clean_1x"], out["T2_rl70_clean_10x"] = r1("R_s70", "1x"), r1("R_s70", "10x")
    if any(arm(n, "1x") is None for n in one):
        out["T3_col5_same"] = out["T4_passes_kept"] = "unread"
    else:
        off = {}
        for n in one:
            t, ls = arm(n, "1x")["rules"]["5"], states[n]["ls_frozen"]["rules"]["5"]  # type: ignore[index]
            if t["width"] is None or t["diff"] is None or ls["diff"] is None:
                off[n] = NO_CALL
            elif E93.r9(abs(t["diff"] - ls["diff"])) > E93.r9(t["width"]):
                off[n] = MISSES
        out["T3_col5_same"] = {"result": NO_CALL if NO_CALL in off.values() else MISSES if off else HOLDS, "off": off}
        base = [n for n in one if states[n]["ls_frozen"]["verdict"] == "passes"]
        lost = [n for n in base if arm(n, "1x")["verdict"] != "passes"]  # type: ignore[index]
        out["T4_passes_kept"] = {"result": NO_CALL if not base else MISSES if lost else HOLDS, "lost": lost, "of": base}
    a1, a10 = arm("R_s70", "1x"), arm("R_s70", "10x")
    if a1 is None or a10 is None:
        out["T5_excess_ratio"] = "unread"
    else:
        e1, e10 = a1["printed"]["excess_share"], a10["printed"]["excess_share"]
        lo, hi = P["thresholds"]["excess_ratio"]
        ratio = e10 / e1 if e1 > 0 else None
        out["T5_excess_ratio"] = {"ratio": ratio, "result": NO_CALL if ratio is None else HOLDS if lo <= ratio <= hi else MISSES}
    return out


def load(name: str) -> dict[str, Any] | None:
    p = RESULTS / name / "state.json"
    return json.loads(p.read_text()) if p.is_file() else None


def score() -> dict[str, Any]:
    P = json.loads((RESULTS / "predictions.json").read_text())
    assert P["thresholds"] == THRESHOLDS and P["first_minutes"] == FIRST, "predictions.json differs from rules.py"
    sp = E93.spreads({n: E93.load(n) for n in E93.STATES})
    states = {n: score_state(E93.load(n), r, sp) for ns in STAGES.values() for n in ns if (r := load(n)) is not None and "shared" in r}
    table = {"states": states, "leans": lean(states, P)}
    tmp = RESULTS / "table.json.tmp"
    tmp.write_text(json.dumps(table, indent=1, default=float) + "\n")
    tmp.replace(RESULTS / "table.json")
    return table


# ----------------------------------------------------------------------------- edge cases, before filing


def fake_state(e93: dict[str, Any], moved_k: float, k11: float, split: dict[str, float], arms: dict[str, Any] | None = None
               ) -> dict[str, Any]:
    """A state.json whose least-squares row reads like `e93` (E93.fake) but with clean moved K `moved_k` and own K11 `k11`."""
    row = lambda x, L: [[x * L, 0.0, 0.0, 0.0, x * L]] * len(e93["n"]["kl"])
    shared = {"clean": {"W0": e93["clean"]["W0"], "FT": e93["clean"]["FT"], "ls_moved": {"K": row(moved_k, 256)}},
              "split": split, "base": e93["base"], "own_frozen": e93["own"] | {"K11": row(k11, 256)}}
    return {"shared": shared, "arms": arms or {}}


def edges() -> bool:
    """Each case names its outcome; E93.fake builds TOFU-row records (ruler 1, do-nothing K01 0.1, rows of 4)."""
    e93 = E93.fake()
    sp = {"anchor 2": (0.0, False)}
    split = {"size": 1.0, "left": 0.1, "moved": 0.9, "null": 0.3}

    def arm_rec(moved_k: float, k11: float, gain: float, left: float = 0.1, r2: float = 1e-4) -> dict[str, Any]:
        row = lambda x, L: [[x * L, 0.0, 0.0, 0.0, x * L]] * 4
        return {"done": True, "clean": {"moved": {"K": row(moved_k, 256)}, "remainder": {"K": row(0.5, 256)}}, "gain": gain,
                "split": split | {"left": left, "moved": 1 - left}, "own": e93["own"] | {"K11": row(k11, 256)},
                "train": {"readings": [{"r2": r2}], "kept_step": 300}}

    def got(state: dict[str, Any]) -> list[str]:
        t = score_state(e93, state, sp)
        rows = [t["ls_frozen"], *t["arms"].values()]
        return [x["verdict"] for x in rows] + [f"1:{x['rules']['1']['result']} 5:{x['rules']['5']['result']}" for x in rows]

    cases: list[tuple[str, Any, Any]] = [
        ("Least squares as E93 read it; trained arm equal to it", got(fake_state(e93, 0.01, 0.1, split, {"1x": arm_rec(0.01, 0.1, 1.2)})),
         ["passes", "passes", "1:holds 5:holds", "1:holds 5:holds"]),
        ("Do-nothing as the trained state (nothing moved): clean K = the ruler, K11 = K01", got(fake_state(e93, 0.01, 0.1, split, {"1x": arm_rec(1.0, 0.1, 3.0, left=1.0)})),
         ["passes", FAILS, "1:holds 5:holds", "1:misses 5:holds"]),
        ("Share at or above its null: printed only, the state still passes", got(fake_state(e93, 0.01, 0.1, split, {"1x": arm_rec(0.01, 0.1, 1.2, left=0.4)})),
         ["passes", "passes", "1:holds 5:holds", "1:holds 5:holds"]),
        ("Column 5 worse than do-nothing (K11 0.2 against 0.1)", got(fake_state(e93, 0.01, 0.1, split, {"1x": arm_rec(0.01, 0.2, 1.2)})),
         ["passes", FAILS, "1:holds 5:holds", "1:holds 5:misses"]),
        ("Clean at the 0.02 line: a tie misses", got(fake_state(e93, 0.02, 0.1, split)), [FAILS, "1:misses 5:holds"]),
        ("Kept clean under t = 0.5's line (column 4 fails)", got(fake_state(e93, 0.01, 0.1, split, {"1x": arm_rec(0.01, 0.1, 2.0)})),
         ["passes", FAILS, "1:holds 5:holds", "1:holds 5:holds"]),
        ("Arm not done: not scored", got(fake_state(e93, 0.01, 0.1, split, {"1x": {"done": False}})), ["passes", "1:holds 5:holds"])]
    v = {"norm": 1.0, "e93": 1.0, "sha256": "a", "e93_sha256": "a"}
    cases += [("norm: sha differs", judge("norm", v | {"sha256": "b"}), False),
              ("card: drift 2e-3", judge("card", {"here": 1.002, "e93": 1.0}), False),
              ("card: NaN", judge("card", {"here": math.nan, "e93": 1.0}), False),
              ("band: zero drift gives the floor", band(0.0), 1e-5),
              ("zero_offsets: r2 nonzero", judge("zero_offsets", {"r2": 1e-12, "tf32_K8": 1.0, "K8": 1.0, "e93_K8": 1.0, "band": 1e-5}), False),
              ("zero_offsets: TF32 off by 2e-3", judge("zero_offsets", {"r2": 0.0, "tf32_K8": 1.002, "K8": 1.0, "e93_K8": 1.0, "band": 1e-5}), False),
              ("first_reading: walk at 0.3 x the forecast holds (tie)", judge("first_reading", {"r2": 0.3, "forecast_r2": 1.0, "median_sqrt_v": 1.0,
               "K8": 1.0, "K8_step0": 1.0, "gpu_gib": 10.0}), True),
              ("first_reading: K8 above step 0", judge("first_reading", {"r2": 1.0, "forecast_r2": 1.0, "median_sqrt_v": 1.0,
               "K8": 1.000001, "K8_step0": 1.0, "gpu_gib": 10.0}), False),
              ("first_reading: √v at Adam's ε scale", judge("first_reading", {"r2": 1.0, "forecast_r2": 1.0, "median_sqrt_v": 1e-8,
               "K8": 1.0, "K8_step0": 1.0, "gpu_gib": 10.0}), False),
              ("gradient: zero on dU", judge("gradient", {"dV": 1.0, "dU": 0.0}), False)]
    P = {"thresholds": THRESHOLDS}
    st = lambda v5, lsv, e1, e10: {"arms": {"1x": {"rules": {"1": {"result": HOLDS}, "5": {"diff": v5, "width": 0.01}}, "verdict": "passes",
                                                   "printed": {"excess_share": e1}},
                                            "10x": {"rules": {"1": {"result": MISSES}}, "printed": {"excess_share": e10}}},
                                   "ls_frozen": {"rules": {"5": {"diff": 0.0}}, "verdict": lsv}}
    every = lambda v5, lsv, e1, e10: {n: st(v5, lsv, e1, e10) for ns in STAGES.values() for n in ns}
    lo = lean(every(0.005, "passes", 1e-4, 1e-2), P)
    cases += [("Leans, three states written: T3, T4 unread", lean({n: st(0, "passes", 1, 1) for n in ("R_s70", "N_s0", "N_s70")}, P)["T3_col5_same"], "unread"),
              ("Leans, every state as least squares: T1 holds, T2 misses, T3 holds, T4 holds, T5 ratio 100 holds",
               [lo["T1_rl70_clean_1x"], lo["T2_rl70_clean_10x"], lo["T3_col5_same"]["result"], lo["T4_passes_kept"]["result"], lo["T5_excess_ratio"]["result"]],
               [HOLDS, MISSES, HOLDS, HOLDS, HOLDS]),
              ("Leans, column 5 moved by twice the width: T3 misses", lean(every(0.02, "passes", 1e-4, 1e-2), P)["T3_col5_same"]["result"], MISSES),
              ("Leans, no least-squares pass: T4 no call", lean(every(0.0, FAILS, 1e-4, 1e-2), P)["T4_passes_kept"]["result"], NO_CALL),
              ("Leans, 1x excess zero (zero denominator): T5 no call", lean(every(0.0, "passes", 0.0, 1e-2), P)["T5_excess_ratio"]["result"], NO_CALL),
              ("plateau: a new low in the last four (2.0 after a best of 3.0)", plateau([3.0, 2.0, 2.0, 2.0, 2.5]), False),
              ("plateau: last four at or above the best before", plateau([3.0, 2.0, 2.1, 2.2, 2.0, 2.3]), True),
              ("plateau: four readings only", plateau([3.0, 3.0, 3.0, 3.0]), False)]
    ok = True
    with (RESULTS / "edges.txt").open("w") as f:
        for c, g, w in cases:
            line = f"{'ok ' if g == w else 'BAD'}  {c}: {g} (named {w})"
            ok &= g == w
            print(line, flush=True)
            f.write(line + "\n")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E111's scoring")
    ap.add_argument("--edges", action="store_true")
    if ap.parse_args().edges:
        sys.exit(0 if edges() else 1)
    t = score()
    print(json.dumps({n: {"ls_frozen": s["ls_frozen"]["verdict"], **{a: x["verdict"] for a, x in s["arms"].items()}}
                      for n, s in t["states"].items()} | {"leans": t["leans"]}, indent=1, default=float), flush=True)
