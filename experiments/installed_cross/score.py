"""E271 (P2.65) scoring, CPU only, over results/read/<base>__<task>_s<seed>_<kind>.json: per reading and kind, the 9
triples (task x seed: does the diagonal state rank above both moved off-diagonal states, in the filed direction) and the
AUC of the 9 diagonal against the 18 off-diagonal states, with the same AUC on their matched Gaussians as the floor.
Rules: results/predictions.json "thresholds.evidence". The released M's row is descriptive (listed, never scored).

    ~/commute-agents/fleet/cpu-run python3 experiments/installed_cross/score.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TASKS, SEEDS = ("addition", "cipher", "yoruba"), (0, 1, 2)


def at(rec: dict[str, Any], path: str) -> float:
    x: Any = rec
    for k in path.split("."):
        x = x.get(k) if isinstance(x, dict) else None
    return float(x) if isinstance(x, (int, float)) and x is not None else math.nan


def signed(v: float, direction: str) -> float:
    return v if direction == "higher" else -v


def triple(d: float, offs: list[float]) -> bool:
    """The diagonal strictly above both off-diagonals; any nan or a tie is not above."""
    return not math.isnan(d) and len(offs) == 2 and all(not math.isnan(o) and d > o for o in offs)


def auc(pos: list[float], neg: list[float]) -> float:
    """P(pos > neg) over finite pairs, ties 1/2; nan when either side is empty."""
    p, n = [x for x in pos if not math.isnan(x)], [x for x in neg if not math.isnan(x)]
    if not p or not n:
        return math.nan
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in p for b in n) / (len(p) * len(n))


def score(vals: dict[tuple[str, str, int], float], ev: dict[str, Any]) -> dict[str, Any]:
    """vals[(base, task, seed)] signed so that higher = reuse; bases are the three moved ones."""
    diag = [vals.get((t, t, s), math.nan) for t in TASKS for s in SEEDS]
    off = [vals.get((b, t, s), math.nan) for t in TASKS for s in SEEDS for b in TASKS if b != t]
    hits = [triple(vals.get((t, t, s), math.nan), [vals.get((b, t, s), math.nan) for b in TASKS if b != t]) for t in TASKS for s in SEEDS]
    fin = {"triples": sum(not math.isnan(vals.get((t, t, s), math.nan)) and all(not math.isnan(vals.get((b, t, s), math.nan)) for b in TASKS if b != t)
                          for t in TASKS for s in SEEDS),
           "diag": sum(not math.isnan(x) for x in diag), "off": sum(not math.isnan(x) for x in off)}  # fmt: skip
    by_task = {t: auc([vals.get((t, t, s), math.nan) for s in SEEDS], [vals.get((b, t, s), math.nan) for s in SEEDS for b in TASKS if b != t])
               for t in TASKS}  # fmt: skip  (reported beside the pooled AUC: a reading's scale may differ by task)
    return {"triples": sum(hits), "of": len(hits), "auc": auc(diag, off), "auc_by_task": by_task, "finite": fin,
            "read": all(fin[k] >= ev["min_finite"][k] for k in fin)}  # fmt: skip


def verdict(real: dict[str, Any], floor: dict[str, Any], ev: dict[str, Any]) -> str:
    """`floor`: the matched Gaussians scored on the reading's floor path (its numerator for a ratio reading, amendment
    2026-11-22); a floor that is not read never passes for one."""
    if not real["read"]:
        return "not read"
    ok = real["triples"] >= ev["triples_min"] and real["auc"] >= ev["auc_min"]
    if not ok:
        return "does not separate"
    if not floor["read"]:
        return "floor not read"
    return "floor separates too" if floor["auc"] >= ev["gauss_auc_max"] else "separates"


def load(kind: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    for p in (RESULTS / "read").glob(f"*_{kind}.json"):
        r = json.loads(p.read_text())
        out[(r["base"], r["task"], r["seed"])] = r
    return out


def main() -> None:
    P = json.loads((RESULTS / "predictions.json").read_text())
    ev = P["thresholds"]["evidence"]
    res: dict[str, Any] = {"experiment": P["experiment"], "kinds": {}}
    gauss = load("gauss")
    for kind in ev["kinds"]:
        recs, rows = load(kind), {}
        for name, (path, direction, floor_path) in ev["readings"].items():
            if name in ev["skip"].get(kind, []):
                continue
            v = {k: signed(at(r, path), direction) for k, r in recs.items() if k[0] != "M"}
            g = {k: signed(at(r, floor_path), direction) for k, r in gauss.items() if k[0] != "M"}
            real, gs = score(v, ev), score(g, ev)
            m_row = {f"{t}_s{s}": at(recs[("M", t, s)], path) for t in TASKS for s in SEEDS if ("M", t, s) in recs}
            rows[name] = {"real": real, "gauss": gs, "verdict": verdict(real, gs, ev), "released_M": m_row}
            print(f"[{kind} {name}] triples {real['triples']}/{real['of']} auc {real['auc']:.3f} | gauss auc {gs['auc']:.3f} -> {rows[name]['verdict']}")
        res["kinds"][kind] = rows
    (RESULTS / "score.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
