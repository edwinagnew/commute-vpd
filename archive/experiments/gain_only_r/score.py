"""P1.41 (E236) score, no GPU: the filed rules H1, R1 and R2 of results/predictions.json and the per-item table, each
gain-only state beside the full fine-tune of the same reference and seed (E226 at VPD, E221 per coarse level, E205's
held model), from the item files present. Writes results/summary.json.

    python experiments/gain_only_r/score.py [--out items]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
X = HERE.parent
E205_ITEMS = X / "held_model_size_matched/results/items"
LEVELS = ("head_mlp", "mlp_attn", "block")
E196_NAMES = {"case_label/real": "case_real", "case_label/twin": "case_twin", "fresh/real": "repeat_real", "fresh/twin": "repeat_twin"}


def fname(item: str) -> str:
    return item.replace("/", "-")


def jload(p: Path) -> Any:
    return json.loads(p.read_text()) if p.is_file() else None


def vpd_R(rec: Any) -> float | None:
    return rec["searches"]["main"]["polished"]["R_read"] if rec and rec.get("reading") == "read" else None


def coarse_R(rec: Any) -> float | None:
    """The mean over the main search's orders, as E221's score.py reads an item."""
    return statistics.fmean(r["R_read"] for r in rec["searches"]["main"]) if rec and rec.get("reading") == "read" else None


def call(d: list[float], th: dict[str, Any]) -> dict[str, Any]:
    """R1/R2: up = items with d > 0 (ties count against rival a); 'too few' under n_min."""
    n, up = len(d), sum(x > 0 for x in d)
    if n < th["n_min"]:
        return {"n": n, "up": up, "call": "too few"}
    c = "rival a" if up >= math.ceil(th["share_a"] * n) else "rival b" if up <= math.floor(th["share_b"] * n) else "no call"
    return {"n": n, "up": up, "call": c, "median_d": statistics.median(d)}


def row(item: str, out: Path) -> dict[str, Any]:
    ref, seed = item.rsplit("_s", 1)
    v, h = jload(out / "vpd" / f"{fname(item)}.json"), jload(out / "held" / f"{fname(item)}.json")
    full = f"{fname(E196_NAMES.get(ref, ref))}_s{seed}"
    r: dict[str, Any] = {"ref": ref, "seed": int(seed), "vpd": vpd_R(v), "kl_fm": v and v["kl_fm"],
                         "vpd_full": vpd_R(jload(X / "kstar_vpd/results/items" / f"{fname(item)}.json")),
                         "held": h and {k: h["readings"][k]["S"] for k in ("1", "2", "3")}}  # fmt: skip
    if v and v.get("reading") == "read":
        p = v["searches"]["main"]["polished"]
        r |= {"f": p["f"], "size": p["size"], "site_remainder_share": p["site_remainder_share"]}
    full_held = jload(E205_ITEMS / f"{fname(item)}.json")
    r["held_full"] = full_held and {k: full_held["readings"][k]["S"] for k in ("1", "3")}
    for lv in LEVELS:
        r[lv] = coarse_R(jload(out / "coarse" / f"{fname(item)}_{lv}.json"))
        r[f"{lv}_full"] = coarse_R(jload(X / "kstar_pairs/results/items" / f"{full}_{lv}.json"))
    return r


def main(out_name: str) -> int:
    pred = json.loads((RESULTS / "predictions.json").read_text())
    th, hand = pred["thresholds"], json.loads((X.parent / pred["settings"]["handoff"]).read_text())
    m = pred["settings"]["match"]
    seed = {st["file"]: re.search(r"_s(\d+)_step\d+\.pt$", st["file"]).group(1) for st in hand["states"].values()}  # type: ignore[union-attr]
    names = sorted({f"{f.split(':', 1)[1]}_s{seed[st['file']]}" for st in hand["states"].values() for f in st["for"] if f.startswith(f"{m}:")})
    out = RESULTS / out_name
    rows = {i: row(i, out) for i in names}
    S1 = {i: r["held"]["1"] if r["held"] else None for i, r in rows.items()}
    h1 = {"n_items": len(S1), "missing_or_null": sorted(i for i, s in S1.items() if s is None),
          "below_bar": sorted(i for i, s in S1.items() if s is not None and s < th["held_bar"]),
          "min_S": min((s for s in S1.values() if s is not None), default=None)}  # fmt: skip
    h1["holds"] = not h1["missing_or_null"] and not h1["below_bar"]

    def d(key: str) -> list[float]:
        return [r[key] - r[f"{key}_full"] for r in rows.values() if r[key] is not None and r[f"{key}_full"] is not None]

    by_task: dict[str, list[float]] = {}
    for r in rows.values():
        if r["vpd"] is not None:
            by_task.setdefault(r["ref"], []).append(r["vpd"])
    summary = {"experiment": pred["experiment"], "H1_held_near_1": h1, "R1_vpd": call(d("vpd"), th),
               "R2_coarse": {lv: call(d(lv), th) for lv in LEVELS},
               "gain_only_seed_sd_vpd": {t: statistics.stdev(v) for t, v in by_task.items() if len(v) > 1}, "items": rows}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("items", "gain_only_seed_sd_vpd")}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="items")
    raise SystemExit(main(ap.parse_args().out))
