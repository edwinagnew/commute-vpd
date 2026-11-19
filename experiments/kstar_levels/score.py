"""E214's bands and filed rules (results/predictions.json "rules": R1, R2, P1-P7) from the item files present, scored
under both grade readings, "read" (the read positions only) and "all" (every position of D, both runs); the main one
is predictions.json "main_grade" (the user, 2026-11-19: "read"). A rule whose items or levels are not all read is "not
read"; one that needs a level the user dropped (predictions.json "stages_run": neuron_head, 2026-11-19) is "not read:
stage dropped by the user", and R2 is read over the levels run. Writes results/summary.json. No GPU:

    python experiments/kstar_levels/score.py
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EDITS = ("reuse_small", "reuse_large", "new_few", "new_thin")
HEADS = ("e201", "last_lowest", "last_rank5", "last_rank9", "last_rank13")
LEVELS = ("neuron_head", "head_mlp", "mlp_attn", "block")
Items = dict[tuple[str, str], dict[str, Any]]
DROPPED_MSG = "not read: stage dropped by the user"


def load() -> Items:
    its = (json.loads(p.read_text()) for p in (RESULTS / "items").glob("*.json") if not p.stem.endswith(".stopped"))
    return {(it["item"], it["level"]): it for it in its if it.get("reading") == "read"}


def recs(d: Items, item: str, L: str, v: str = "main") -> list[dict[str, Any]]:
    return d[(item, L)]["searches"][v]  # KeyError: not read


def R(d: Items, item: str, L: str, g: str, v: str = "main") -> float:
    return statistics.mean(r[f"R_{g}"] for r in recs(d, item, L, v))


def band(d: Items, L: str, g: str, k: float) -> float:
    """k x the largest, over the four edits, of the sd of R over orders, |R(half 1) - R(half 2)| and R's tie range."""
    spread = []
    for e in EDITS:
        m = [r[f"R_{g}"] for r in recs(d, e, L)]
        spread += [statistics.stdev(m) if len(m) > 1 else 0.0, abs(R(d, e, L, g, "half1") - R(d, e, L, g, "half2")), recs(d, e, L)[0].get(f"R_{g}_tie_range", 0.0)]
    return k * max(spread)


def gap(d: Items, L: str, g: str, v: str = "main") -> float:
    return min(R(d, "reuse_small", L, g, v), R(d, "reuse_large", L, g, v)) - max(R(d, "new_few", L, g, v), R(d, "new_thin", L, g, v))


def recovers(d: Items, L: str, g: str, b: float) -> bool:
    return gap(d, L, g) > b and all(statistics.mean(r["f"] for r in recs(d, e, L)) > 0 for e in ("reuse_small", "reuse_large"))


def rule(f: Callable[[], Any]) -> Any:
    try:
        return f()
    except KeyError:
        return "not read"


def scored(d: Items, g: str, th: dict[str, Any], dropped: tuple[str, ...]) -> dict[str, Any]:
    tol, k = th["s_tol"], th["band_factor"]
    b: dict[str, Any] = {L: DROPPED_MSG if L in dropped else rule(lambda L=L: band(d, L, g, k)) for L in LEVELS}
    r1 = {L: b[L] if isinstance(b[L], str) else rule(lambda L=L: "recovers" if recovers(d, L, g, b[L]) else "does not") for L in LEVELS}
    out: dict[str, Any] = {"band": b, "gap": {L: rule(lambda L=L: gap(d, L, g)) for L in LEVELS}, "R1": r1}
    run = [r1[L] for L in LEVELS if L not in dropped]  # R2 over the levels run
    out["R2"] = "not read" if "not read" in run else "pass" if "recovers" in run else "fail"
    out["passing_levels"] = [L for L, v in r1.items() if v == "recovers"]
    out["R"] = {f"{i}@{L}": R(d, i, L, g) for (i, L) in d}
    nh, hm = "neuron_head", "head_mlp"
    out["P1"] = rule(lambda: R(d, "new_thin", nh, g) >= th["thin_floor"] and R(d, "new_thin", nh, g) > R(d, "new_few", nh, g) + b[nh] and r1[nh] == "does not")
    out["P2"] = r1[hm] if r1[hm] == "not read" else r1[hm] == "recovers"
    out["P3"] = rule(lambda: all(r1[L] == "does not" and R(d, "reuse_large", L, g) <= max(R(d, "new_few", L, g), R(d, "new_thin", L, g)) + b[L] for L in ("mlp_attn", "block")))
    out["P4"] = rule(lambda: all(abs(R(d, x, nh, g) - 1) <= tol and all(abs(r[f"rho_{g}"] - 1) <= tol for r in recs(d, x, nh) if r["f"] > 0) for x in ("x2", "x10"))
                     and all(R(d, "x2", L, g) - R(d, "x10", L, g) > b[L] for L in LEVELS[1:]))  # fmt: skip
    out["P5"] = rule(lambda: all(all(r["lacked_in_K"] >= 1 and (r["changed_in_K"] > 1 or r[f"rho_{g}"] == 0) for r in recs(d, h, L)) for h in HEADS for L in (nh, hm))
                     and all(R(d, h, hm, g) < R(d, "reuse_small", hm, g) - b[hm] for h in HEADS))  # fmt: skip
    out["P6"] = rule(lambda: R(d, "moved", nh, g) < 1 - tol and all(r["lacked_in_K"] >= 1 for r in recs(d, "moved", nh))
                     and all(abs(R(d, "moved", L, g) - 1) <= tol for L in LEVELS[1:]))  # fmt: skip
    out["P7"] = rule(lambda: {L: {v: gap(d, L, g, v) for v in ("alpha0.8", "alpha0.95", "restricted")} for L in out["passing_levels"]}) if out["passing_levels"] else "not read"
    if isinstance(out["P7"], dict):
        out["P7_holds"] = all(x > 0 for v in out["P7"].values() for x in v.values())
    if nh in dropped:
        out |= dict.fromkeys(("P1", "P4", "P5", "P6"), DROPPED_MSG)  # each reads neuron_head
    return out


def main() -> None:
    pred = json.loads((RESULTS / "predictions.json").read_text())
    dropped = ("neuron_head",) if "neuron_head dropped" in pred.get("stages_run", "") else ()
    d, main_g = load(), pred["main_grade"]
    out = {"experiment": pred["experiment"], "main_grade": main_g, "main_grade_source": pred["main_grade_source"], "stages_run": pred.get("stages_run"),
           "read": sorted(f"{i}@{L}" for i, L in d)} | {f"grade_{g}": scored(d, g, pred["thresholds"], dropped) for g in ("all", "read")}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"main grade: {main_g} (side: {'all' if main_g == 'read' else 'read'}); stages run: {pred.get('stages_run')}")
    for g in (main_g, *(x for x in ("all", "read") if x != main_g)):
        s = out[f"grade_{g}"]
        print(f"grade {g} ({'main' if g == main_g else 'side'}): R1 {s['R1']}; R2 {s['R2']}; " + ", ".join(f"{p} {s[p]}" for p in ("P1", "P2", "P3", "P4", "P5", "P6")) + f"; P7 {s.get('P7_holds', s['P7'])}")


if __name__ == "__main__":
    main()
