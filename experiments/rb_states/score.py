"""E257 (cell P1.50) summary of the items stage from the saved item JSONs only (no model, no GPU): stage 0, the two
random updates, every item's E_b, untouched share, |K*|, ||dW|| / ||W0|| and R under wording A where E242/E248 saved
it; per E187 pair (E242's pair_sides) the seed means of each side, the gap reuse minus new and whether the reuse side
is higher, also against update size; E62's copy/lookup at own size; E237's key tasks. Descriptive, no rule (P1.1).
results/summary.json holds that under "items" and, per arm, the rl stage (E220 size-matched, E218 KL-matched) and
the manifest stage (P2.56 procedural states): per seed E_b, untouched share, |K*| and status, and the arm's means over
its usable seeds; plus how many K* searches hit the evaluation cap (from the stage logs).

    ~/commute-agents/fleet/cpu-run python3 experiments/rb_states/score.py   (writes results/summary.json, summary_items.json)
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


def arms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows {arm, seed, status, E_b, untouched_share, size_K} grouped by arm; means over the usable seeds only."""
    out: dict[str, Any] = {}
    for a in dict.fromkeys(r["arm"] for r in rows):
        mine = sorted((r for r in rows if r["arm"] == a), key=lambda r: r["seed"])
        ok = [r for r in mine if r["status"] == "usable"]
        out[a] = {"n": len(mine), "n_usable": len(ok), "E_b": fmean(r["E_b"] for r in ok) if ok else None,
                  "untouched_share": fmean(r["untouched_share"] for r in ok) if ok else None,
                  "seeds": {str(r["seed"]): {k: r[k] for k in ("status", "E_b", "untouched_share", "size_K")} for r in mine}}  # fmt: skip
    return out


def stage_rows(stage: str) -> list[dict[str, Any]]:
    """Every read item JSON of the rl or manifest stage, with its arm ("E220 rl", "addition sft") and seed."""
    man = {e["item"]: e for e in json.loads((RESULTS / "manifest_p256.json").read_text())}
    rows = []
    for f in sorted((RESULTS / "items").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("stage") != stage:
            continue
        assert d["reading"] == "read", d["item"]
        name, seed = d["item"].rsplit("_s", 1)
        if stage == "rl":  # rl/stage2b/repeat_rl_s0: stage2b = E220 size-matched, stage2 = E218 KL-matched
            arm = f"{'E220' if '/stage2b/' in name else 'E218'} {name.rsplit('/repeat_', 1)[1]}"
        else:
            arm = f"{man[d['item']]['task']} {man[d['item']]['arm']}"
        rows.append(d | {"arm": arm, "seed": int(seed), "status": d.get("status", "usable")})
    return rows


def searches() -> dict[str, int]:
    """K* searches of the rl and manifest stages (one 'main:' line each in the logs) and how many hit the cap."""
    lines = [ln for f in sorted(RESULTS.glob("run_*.log")) for ln in f.read_text().splitlines() if " main: |K*|" in ln]
    return {"n": len({ln.split(" main:")[0].strip() for ln in lines if " rl/" in ln or " p256/" in ln}),
            "cap_hit": sum("cap hit" in ln for ln in lines if " rl/" in ln or " p256/" in ln)}


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
    man = stage_rows("manifest")
    bad = [r["E_b"] for r in man if r["status"] != "usable"]
    full = {"experiment": out["experiment"], "cell": out["cell"], "reading": out["reading"], "items": out,
            "rl": arms(stage_rows("rl")), "manifest": arms(man),
            "manifest_not_usable": {"n": len(bad), "E_b_min": min(bad), "E_b_max": max(bad)}, "kstar_searches": searches()}  # fmt: skip
    (RESULTS / "summary.json").write_text(json.dumps(full, indent=1))
    short = {p: (round(v["reuse"]["E_b"], 4), round(v["new"]["E_b"], 4), round(v["gap_E_b"], 4), round(v["size_ratio_reuse_over_new"], 3)) for p, v in pairs.items()}
    print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "stage0", "keys", "e62_own_size", "randoms")} | {"pairs": short}, indent=1))


if __name__ == "__main__":
    main()
