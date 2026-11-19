"""Cell P3.13 run 1 (E216): the description from the written records, no GPU (`run.py --summary` calls `summarise`).
Per state: S and its random change's S, the share placed on the random curve (`share_reading.place`), effective rank and
its floor. The filed rules (results/predictions.json "rules") are `s_rule` and `share_class`. No reuse or new label.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from commute import qwen_reading as qr
from commute import share_reading as SR

HERE = Path(__file__).resolve().parent
Get = Callable[[str], dict[str, Any]]
LEVELS = ("low", "mid", "high")


def s_rule(gaps: list[float | None], smallest: float) -> str:
    """The card's rule on one run's seeds, gap = S(state) - S(its random change): 'above random' when every gap >
    smallest, 'below random' when every gap < -smallest; 'like random' when every |gap| <= smallest, or the gaps have
    both signs; 'mixed' otherwise; 'unreadable' when a gap is missing (an S 'no change') or there are none. Rounded to
    9 places."""
    if not gaps or any(g is None or not qr.ok(g) for g in gaps):
        return "unreadable"
    gs = [qr.r9(float(g)) for g in gaps if g is not None]
    if all(g > qr.r9(smallest) for g in gs):
        return "above random"
    if all(g < -qr.r9(smallest) for g in gs):
        return "below random"
    if all(abs(g) <= qr.r9(smallest) for g in gs) or (any(g > 0 for g in gs) and any(g < 0 for g in gs)):
        return "like random"
    return "mixed"


def share_class(arm: str, rec: dict[str, Any], cv: list[tuple[float, float]], band: float | None) -> dict[str, Any]:
    """The state's share against the curve; an unscored own-samples SFT share reads 'does not apply' (lead, flag 1),
    any other unscored share 'unreadable'."""
    scored = rec["status"] == "scored"
    if not scored and arm == "F":
        return {"curve": None, "gap": None, "band": band, "class": "does not apply"}
    return SR.place(rec["regating"], scored, rec["means"]["K00"], cv, band)


def curve(get: Get, draws: list[int], match: float) -> tuple[list[tuple[float, float]], float | None, list[Any]]:
    recs = [(d, lv, get(f"curve/d{d}_{lv}")) for d in draws for lv in LEVELS]
    pts = [{"draw": d, "level": lv, "k00": r["means"]["K00"], "share": r["regating"],
            "usable": r["status"] == "scored" and qr.within(r["means"]["K00"], r["target_k00"], match)}
           for d, lv, r in recs if r]
    cv = SR.curve(pts)
    return cv, SR.band(cv, pts), pts


def sd(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def state_row(name: str, get: Get, cv: list[tuple[float, float]], band: float | None) -> dict[str, Any]:
    s, sr, sh, rk = get(f"S/{name}"), get(f"S/{name}_random"), get(f"share/{name}"), get(f"rank/{name}")
    row: dict[str, Any] = {"arm": name.split("_s")[0], "seed": int(name.split("_s")[1])}
    row |= {"S": s.get("S"), "kl_fm": s.get("kl_fm"), "S_random": sr.get("S"), "kl_fm_random": sr.get("kl_fm")}
    row["S_gap"] = None if row["S"] is None or row["S_random"] is None else row["S"] - row["S_random"]
    if sh:
        row |= {"share": sh["regating"], "share_status": sh["status"], "k00": sh["means"]["K00"]}
        row |= {("share_class" if k == "class" else k): v for k, v in share_class(row["arm"], sh, cv, band).items()}
    else:
        row["share_class"] = "not read"
    row |= {"erank": rk.get("erank"), "erank_floor": rk.get("floor")}
    return row


def summarise(C: Any, get: Get) -> dict[str, Any]:
    cv, band, pts = curve(get, list(C.share_draws), C.match)
    states = {n: state_row(n, get, cv, band) for n in C.states}
    smallest = json.loads((HERE / "results" / "predictions.json").read_text())["rules"]["S"]["smallest_gap"]
    arms = sorted({r["arm"] for r in states.values()}, key="RNF".index)
    per_arm = {a: {"S_gaps": [r["S_gap"] for r in states.values() if r["arm"] == a],
                   "S_seed_sd": sd([r["S"] for r in states.values() if r["arm"] == a and r["S"] is not None])}
               for a in arms}
    for v in per_arm.values():
        v["S_rule"] = s_rule(v["S_gaps"], smallest)  # filed for RL; the others are described
    out = {"experiment": "E216", "cell": "P3.13", "states": states, "per_arm": per_arm,
           "curve": {"points": cv, "band": band, "draws": pts, "skipped": bool(get("curve/skipped"))},
           "share_p9707d6ff": "not read for these states (lead, flag 4)"}
    write_json(HERE / "results" / "summary.json", out)
    for n, r in states.items():
        print(f"[summary] {n}: {json.dumps(r, default=float)}", flush=True)
    print(f"[summary] per arm {json.dumps(per_arm, default=float)}; band {band}", flush=True)
    return out


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)
