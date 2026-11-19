"""E213 (cell P1.34, test T1): bounds on the definition's reuse score R from E208's saved per-part scores.

Every neuron and head of these full fine-tunes is changed, so R = mean of s_u over the behaviour's circuit K*. With
K* unknown, read for each item and distance: the range of R (lowest and highest single part), the worst R for a
circuit of k parts (mean of the k lowest s_u), and R on E208's five searched circuits; then reuse-minus-twin gaps.

    ~/commute-agents/fleet/cpu-run python experiments/r_bounds_e208/run.py
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from commute.jobs import config_from_settings, hard_exit, require_filed

HERE = Path(__file__).parent
REPO = HERE.parents[1]


@dataclass(frozen=True)
class Config:
    parts_dir: str
    items_dir: str
    distances: list[str]
    k_grid: list[int]
    k_min: int
    level_bar: float
    gap_bar: float
    pairs: list[list[str]]
    counted: int
    seeds: list[int]
    orders: int
    built: list[str]
    check_tol: float


def read_item(cfg: Config, name: str) -> dict:
    """Range, worst-k curve and R on E208's circuits for one item, per distance. 'all' is every part."""
    z = np.load(Path(cfg.parts_dir) / f"{name}.npz")
    out: dict = {"n_parts": int(z["parts"].size)}
    for d in cfg.distances:
        s = np.sort(z[f"r_{d}"], kind="stable")
        ks = sorted({min(k, s.size) for k in cfg.k_grid})
        csum = np.cumsum(s)
        Ks = [z[f"K_replace_{j}"] for j in range(cfg.orders)]
        out[d] = {"min": float(s[0]), "max": float(s[-1]),
                  "worst_k": {str(k): float(csum[k - 1] / k) for k in ks} | {"all": float(s.mean())},
                  "e208_K": [float(z[f"r_{d}"][K].mean()) for K in Ks],
                  "e208_K_size": [int(K.size) for K in Ks]}
        out[d]["e208_K_mean"] = float(np.mean(out[d]["e208_K"]))
    return out


def check_against_e208(cfg: Config, name: str, got: dict) -> list[str]:
    """R0: the npz must reproduce E208's own summaries (k = all and each order's rho)."""
    e = json.loads((REPO / cfg.items_dir / f"{name}.json").read_text())
    miss = []
    for d in cfg.distances:
        pairs = [("all", got[d]["worst_k"]["all"], e["r_changed"][d]["mean"])]
        pairs += [(f"order {j}", got[d]["e208_K"][j], o[f"rho_{d}"]) for j, o in enumerate(e["replace"]["orders"])]
        miss += [f"{name} {d} {w}: {a} vs {b}" for w, a, b in pairs if abs(a - b) > cfg.check_tol]
    return miss


def gaps(cfg: Config, items: dict) -> list[dict]:
    """Reuse member minus its twin, per pair, seed, distance and k (and E208's circuits, mean over orders)."""
    rows = []
    for p, (reuse, new) in enumerate(cfg.pairs):
        for d in cfg.distances:
            for k in [*map(str, cfg.k_grid), "all", "e208_K"]:
                g = []
                for seed in cfg.seeds:
                    a, b = items[f"{reuse}_s{seed}"][d], items[f"{new}_s{seed}"][d]
                    g.append(a["e208_K_mean"] - b["e208_K_mean"] if k == "e208_K" else a["worst_k"][k] - b["worst_k"][k])
                rows.append({"pair": f"{reuse}/{new}", "counted": p < cfg.counted, "distance": d, "k": k,
                             "gap": g, "seed_diff": g[0] - g[1]})
    return rows


def at_or_above_kmin(cfg: Config, k: str) -> bool:
    return k in ("all", "e208_K") or int(k) >= cfg.k_min


def verdict(cfg: Config, items: dict, rows: list[dict]) -> dict:
    trained = [f"{a}_s{s}" for pair in cfg.pairs for a in pair for s in cfg.seeds]
    low = [(n, d, k, v) for n in trained for d in cfg.distances for k, v in items[n][d]["worst_k"].items()
           if at_or_above_kmin(cfg, k) and v < cfg.level_bar]
    counted = [r for r in rows if r["counted"] and at_or_above_kmin(cfg, r["k"])]
    wide = [r for r in counted if max(abs(g) for g in r["gap"]) >= cfg.gap_bar]
    b_both = [r for r in counted if r["distance"] == "b" and r["k"] != "e208_K"
              and min(abs(g) for g in r["gap"]) >= cfg.gap_bar]
    a_holds, b_holds = not low and not wide, bool(b_both)
    assert not (a_holds and b_holds)
    lowest = {d: min((items[n][d]["worst_k"][str(cfg.k_min)], n) for n in trained) for d in cfg.distances}
    return {"verdict": "A" if a_holds else "B" if b_holds else "neither",
            "A_level_misses": len(low), "A_level_misses_by_distance": {d: sum(x[1] == d for x in low) for d in cfg.distances},
            "lowest_worst_R_at_k_min": {d: {"item": n, "R": v} for d, (v, n) in lowest.items()},
            "A_gap_misses": [{k: r[k] for k in ("pair", "distance", "k", "gap")} for r in wide],
            "B_hits": [{k: r[k] for k in ("pair", "k", "gap", "seed_diff")} for r in b_both]}


def main() -> None:
    pred = HERE / "results" / "predictions.json"
    require_filed(pred)
    cfg = config_from_settings(Config, json.loads(pred.read_text())["settings"])
    trained = [f"{a}_s{s}" for pair in cfg.pairs for a in pair for s in cfg.seeds]
    items = {n: read_item(cfg, n) for n in trained + cfg.built}
    miss = [m for n in trained for m in check_against_e208(cfg, n, items[n])]
    if miss:
        print("R0 missed:\n" + "\n".join(miss[:20]))
        hard_exit(1)
    print(f"R0 held: {len(trained)} items x {len(cfg.distances)} distances match E208's summaries within {cfg.check_tol}")
    rows = gaps(cfg, items)
    out = {"items": items, "gaps": rows, "verdict": verdict(cfg, items, rows)}
    for d in cfg.distances:
        mins = [items[n][d]["min"] for n in trained]
        print(f"{d}: single-part min {min(mins):.4f}-{max(mins):.4f}; "
              + " ".join(f"k={k}:{min(items[n][d]['worst_k'][k] for n in trained):.4f}" for k in ("1", "16", "256", "4096", "all")))
    for r in rows:
        if r["counted"] and at_or_above_kmin(cfg, r["k"]) and r["k"] in ("16", "256", "4096", "all", "e208_K"):
            print(f"gap {r['pair']:22s} {r['distance']:4s} k={r['k']:6s} {r['gap'][0]:+.5f} {r['gap'][1]:+.5f}")
    print(json.dumps(out["verdict"], indent=1))
    (HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    hard_exit(0)


if __name__ == "__main__":
    main()
