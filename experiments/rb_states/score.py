"""E257 (cell P1.50) summary of the items stage from the saved item JSONs only (no model, no GPU): stage 0, the two
random updates, every item's E_b, untouched share, |K*|, ||dW|| / ||W0|| and R under wording A where E242/E248 saved
it; per E187 pair (E242's pair_sides) the seed means of each side, the gap reuse minus new and whether the reuse side
is higher, also against update size; E62's copy/lookup at own size; E237's key tasks. Descriptive, no rule (P1.1).
The RL stage adds its own key later; this file is not results/summary.json.

    ~/commute-agents/fleet/cpu-run python3 experiments/rb_states/score.py      (writes results/summary_items.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.jobs import EXPERIMENTS, load_experiment

E242: Any = load_experiment("kstar_vpd_wording_a", "run")  # pair_sides, key_items, the e62 items, fname
RESULTS = Path(__file__).resolve().parent / "results"
BUILT = EXPERIMENTS / "option_b_rescore" / "results" / "summary.json"  # P1.49's E_b on the built edits, the reference
FIELDS = ("E_b", "untouched_share", "size_K", "n_changed_in_K", "n_grafts_in_K", "dw_rel_norm", "R_A_saved")


def item(name: str) -> dict[str, Any]:
    d = json.loads((RESULTS / "items" / f"{E242.fname(name)}.json").read_text())
    assert d["reading"] == "read", name
    return {"item": name} | {k: d[k] for k in FIELDS}


def side(names: list[str]) -> dict[str, Any]:
    rows = [item(n) for n in names]
    mean = lambda k: fmean(r[k] for r in rows if r[k] is not None) if any(r[k] is not None for r in rows) else None
    return {"n": len(rows), "E_b": mean("E_b"), "E_b_min": min(r["E_b"] for r in rows), "E_b_max": max(r["E_b"] for r in rows),
            "dw_rel_norm": mean("dw_rel_norm"), "R_A_saved": mean("R_A_saved"), "untouched_share": mean("untouched_share"),
            "items": rows}  # fmt: skip


def pair(r: list[str], n: list[str]) -> dict[str, Any]:
    a, b = side(r), side(n)
    gap = a["E_b"] - b["E_b"]
    return {"reuse": a, "new": b, "gap_E_b": gap, "reuse_higher": gap > 0, "size_ratio_reuse_over_new": a["dw_rel_norm"] / b["dw_rel_norm"],
            "gap_R_A_saved": a["R_A_saved"] - b["R_A_saved"]}  # fmt: skip


def main() -> None:
    cfg, _ = E242.load_config()
    stage0 = json.loads((RESULTS / "stage0_p0.json").read_text())
    randoms = {i: item(i) for i in ("random_size_241001", "random_kl_241002")}
    pairs = {p: pair(r, n) for p, (r, n) in E242.pair_sides().items()}
    keys = E242.key_items()
    tasks = sorted({i.split("/")[1].removesuffix("_scaled").rsplit("_s", 1)[0] for i in keys})
    built = json.loads(BUILT.read_text())["items"]
    every = {r["item"]: r for p in pairs.values() for s in ("reuse", "new") for r in p[s]["items"]}
    every |= {i: item(i) for i in (*keys, *cfg.e62)}
    gaps = [p["gap_E_b"] for p in pairs.values()]
    out = {"experiment": "E257 (rb_states)", "cell": "P1.50", "stage": "items",
           "reading": "E_b as filed in predictions.json; descriptive, no verdict under either wording",
           "stage0": stage0, "randoms": randoms,
           "built_reference_P1_49": {k: {"E_b": v["E_b"], "untouched_share": v["untouched_share"]} for k, v in built.items()},
           "pairs": pairs,
           "pairs_reuse_higher": sum(p["reuse_higher"] for p in pairs.values()), "pairs_n": len(pairs),
           "pair_gap_min": min(gaps), "pair_gap_max": max(gaps), "pair_gap_mean": fmean(gaps),
           "pairs_gap_above_random": sum(g > max(r["E_b"] for r in randoms.values()) for g in gaps),
           "pairs_gap_below_minus_random": sum(g < -max(r["E_b"] for r in randoms.values()) for g in gaps),
           "e62_own_size": pair([i for i in cfg.e62 if "/copy" in i], [i for i in cfg.e62 if "/lookup" in i]),
           "keys": {t: side([i for i in keys if f"/{t}_s" in i]) for t in tasks},
           "trained": {"n_items": len(every), "E_b_min": min(r["E_b"] for r in every.values()), "E_b_max": max(r["E_b"] for r in every.values()),
                       "untouched_max": max(r["untouched_share"] for r in every.values()),
                       "grafts_max": max(r["n_grafts_in_K"] for r in every.values())}}  # fmt: skip
    (RESULTS / "summary_items.json").write_text(json.dumps(out, indent=1))
    short = {p: (round(v["reuse"]["E_b"], 4), round(v["new"]["E_b"], 4), round(v["gap_E_b"], 4), round(v["size_ratio_reuse_over_new"], 3)) for p, v in pairs.items()}
    print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "stage0", "keys", "e62_own_size", "randoms")} | {"pairs": short}, indent=1))


if __name__ == "__main__":
    main()
