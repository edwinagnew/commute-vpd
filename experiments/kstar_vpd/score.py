"""P1.37's band and filed rules (results/predictions.json "rules") from the item files present. No GPU:

    python experiments/kstar_vpd/score.py            (writes results/summary.json)

An item's R is the main grade ("read") at its polished K*, all of D. The pairs' rule is E221's T4 (`kstar_pairs/score.py`
tally, spearman), called and not changed, with E221's pair list and E205's saved held-model readings.
"""

from __future__ import annotations

import json
from typing import Any

from commute.jobs import load_experiment

RUN: Any = load_experiment("kstar_vpd", "run")  # load_config, RESULTS, KP (E221's run)
SP: Any = load_experiment("kstar_pairs", "score")  # gap, tally, old_readings
RESULTS = RUN.RESULTS
R1B_REUSE = ("x2", "x10", "rebuilt")


def load_items(out: str = "items") -> dict[str, dict[str, Any]]:
    its = (json.loads(p.read_text()) for p in (RESULTS / out).glob("*.json") if not p.stem.endswith(".stopped"))
    return {it["item"]: it for it in its if it.get("reading") == "read"}


def R(d: dict[str, dict[str, Any]], item: str, variant: str = "main", grade: str = "read") -> float | None:
    it = d.get(item)
    return None if it is None or variant not in it["searches"] else it["searches"][variant]["polished"][f"R_{grade}"]


def band(d: dict[str, dict[str, Any]], known: tuple[str, ...], band_min: float) -> dict[str, Any]:
    diffs = {i: abs(a - b) for i in known if (a := R(d, i, "half1")) is not None and (b := R(d, i, "half2")) is not None}
    value = max(diffs.values(), default=0.0)
    return {"per_item": diffs, "band": value if value > 0 else band_min, "degenerate": value == 0, "n_items": len(diffs)}


def recovered(d: dict[str, dict[str, Any]], reuse: tuple[str, ...], new: tuple[str, ...], b: float) -> dict[str, Any]:
    """min R over `reuse` - max R over `new` against the band; 'not read' unless every item is read."""
    vals = {i: R(d, i) for i in (*reuse, *new)}
    if any(v is None for v in vals.values()):
        return {"call": "not read", "missing": [i for i, v in vals.items() if v is None]}
    lo, hi = min(float(vals[i]) for i in reuse), max(float(vals[i]) for i in new)  # type: ignore[arg-type]
    return {"R": vals, "min_reuse": lo, "max_new": hi, "gap": lo - hi, "holds": lo - hi > b}


def r1b_call(rec: dict[str, Any], d: dict[str, dict[str, Any]]) -> str:
    if "gap" not in rec:
        return "not read"
    if rec["holds"]:
        return "rival a: R1b recovers"
    thin = R(d, "new_thin")
    return "rival b: new_thin at or above the reuse items" if thin is not None and thin >= rec["min_reuse"] else "no call"


def moved(d: dict[str, dict[str, Any]], b: float, s_tol: float) -> dict[str, Any]:
    it = d.get("moved")
    if it is None:
        return {"call": "not read"}
    p = it["searches"]["main"]["polished"]
    others = [R(d, i) for i in R1B_REUSE]
    out = {"R": p["R_read"], "graft_share": p["graft_share"], "identity_holds": p["R_read"] <= 1 - p["graft_share"] + s_tol}
    if all(v is not None for v in others):
        out["P_moved2_holds"] = p["R_read"] < min(float(v) for v in others if v is not None) - b
    return out


def pairs_rule(d: dict[str, dict[str, Any]], th: dict[str, Any]) -> dict[str, Any]:
    """E221's T4 on E187's nine pairs: R gap against minus the update-size gap and against the held-model S3 gap."""
    cfg = RUN.KP.load_config()[0]
    pairs = {p: v for p, v in RUN.KP.pairs(cfg).items() if p.startswith("e187_")}
    g = {k: {p: SP.gap(lambda i, k=k: R(d, i, grade=k), r, n) for p, (r, n) in pairs.items()} for k in ("read", "all")}

    def size(i: str) -> float | None:
        return -d[i]["size"]["whole"] if i in d else None

    minus_size = {p: SP.gap(size, r, n) for p, (r, n) in pairs.items()}
    s3 = {p: v for p, v in SP.old_readings(cfg, RUN.KP.pairs(cfg))["S3_held_model"]["gaps"].items() if p in pairs}
    return {"gaps": g, "minus_size_gap": minus_size, "S3_gap": s3,
            "vs_minus_size": SP.tally(g["read"], minus_size, minus_size, th), "vs_S3": SP.tally(g["read"], s3, minus_size, th)}  # fmt: skip


def main() -> int:
    cfg, th = RUN.load_config()
    d = load_items()
    b = band(d, cfg.known, th["band_min"])
    r1, r1b = recovered(d, cfg.reuse, cfg.new, b["band"]), recovered(d, R1B_REUSE, cfg.new, b["band"])
    out = {"experiment": RUN.EXPERIMENT, "n_items": len(d), "band": b, "R1": r1, "R1b": r1b | {"call": r1b_call(r1b, d)},
           "moved": moved(d, b["band"], th["s_tol"]), "pairs": pairs_rule(d, th),
           "per_item": {i: {"R_read": R(d, i), "R_all": R(d, i, grade="all"), **{k: it["searches"]["main"]["polished"][k] for k in ("f", "size", "site_remainder_share", "graft_share")},
                            "prefix_size": it["searches"]["main"]["prefix"]["size"], "capped": it["searches"]["main"]["capped"]} for i, it in d.items()}}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_items", "band", "R1", "R1b", "moved")}, indent=1))
    print(f"T4 vs minus size: {out['pairs']['vs_minus_size']['call']}; vs S3: {out['pairs']['vs_S3']['call']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
