"""Cell P1.42: R recomputed from E226's saved per-variable grades, plus saved simple readings. CPU, saved files only.

    ~/commute-agents/fleet/cpu-run --mem 4G python experiments/r_reanalysis/run.py

Plan: reports/plans/2026-11-20_plan_p1-42-r-reanalysis.md. Writes results/summary.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
VPD = ROOT / "experiments/kstar_vpd/results"
PAIRS_JSON = ROOT / "experiments/kstar_pairs/results/pairs.json"
E232 = ROOT / "experiments/kstar_size_matched/results/summary.json"
E197 = ROOT / "experiments/held_model_retention/results/summary.json"
E231 = Path.home() / "commute-agents/compare-lead/experiments/gain_only/results/score.json"
OUT = Path(__file__).resolve().parent / "results"

REUSE = ("x2", "x10", "moved", "rebuilt")
NEW = ("new_few", "new_thin")
KNOWN = (*REUSE, *NEW)
MEASURES = ("R", "Rw_K", "Rw_all")


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def npz_path(item: str) -> Path:
    name = "stage0_pod0_do_nothing" if item == "do_nothing" else "items_" + item.replace("/", "-")
    return VPD / "large" / f"{name}.npz"


def wmean(s: np.ndarray, w: np.ndarray, mask: np.ndarray) -> float:
    w = np.abs(w.astype(np.float64)) * mask
    return float((w * s).sum() / w.sum())


def readings(item: str) -> dict[str, Any]:
    """Measures 1-3 on all of D (main rows), their side rows, and for the built edits the two halves of D."""
    z = np.load(npz_path(item))
    s_read, s_all = z["s"][1].astype(np.float64), z["s"][0].astype(np.float64)
    k, w = z["main_in_K_polished"], z["main_attribution"]
    every = np.ones_like(k)
    out = {"R": float(s_read[k].mean()), "Rw_K": wmean(s_read, w[0], k), "Rw_all": wmean(s_read, w[0], every),
           "side": {"Rw_K_allpos": wmean(s_all, w[0], k), "Rw_K_prefix_weight": wmean(s_read, w[2], k),
                    "Rw_all_allpos": wmean(s_all, w[0], every)}}  # fmt: skip
    if "half1_in_K_polished" in z.files:
        out["halves"] = {m: [f(s_read, z[f"half{h}_attribution"], z[f"half{h}_in_K_polished"]) for h in (1, 2)]
                         for m, f in (("R", lambda s, w, k: float(s[k].mean())), ("Rw_K", lambda s, w, k: wmean(s, w[0], k)),
                                      ("Rw_all", lambda s, w, k: wmean(s, w[0], np.ones_like(k))))}  # fmt: skip
    return out


def item_json(item: str) -> dict[str, Any]:
    return load(VPD / "items" / f"{item.replace('/', '-')}.json")


def sizes(items: list[str]) -> dict[str, float]:
    """||dW|| / ||W0|| over the 24 matrices: saved for E187 states; for the built edits from the split checks' dW2."""
    js = {i: item_json(i) for i in items}
    ratios = [sum(c["dW2"] for c in j["checks"].values()) / j["size"]["whole"] ** 2 for j in js.values() if "size" in j]
    w0sq = float(np.median(ratios))
    assert max(ratios) / min(ratios) - 1 < 1e-6, "||W0||^2 differs across E187 states"
    return {i: j["size"]["whole"] if "size" in j else math.sqrt(sum(c["dW2"] for c in j["checks"].values()) / w0sq) for i, j in js.items()}


def size_removed(r: dict[str, dict[str, Any]], size: dict[str, float], m: str) -> tuple[dict[str, float], list[float]]:
    """Minus the residual of log(1 - R) on log size, fitted over every changed state."""
    items = [i for i in size if size[i] > 0 and r[i][m] < 1]
    x, y = np.log([size[i] for i in items]), np.log([1 - r[i][m] for i in items])
    b, a = np.polyfit(x, y, 1)
    return {i: float(-(yi - (a + b * xi))) for i, xi, yi in zip(items, x, y, strict=True)}, [float(a), float(b)]


def slope_window(r: dict[str, dict[str, Any]], size: dict[str, float], m: str) -> dict[str, Any]:
    """The slopes b (step 0.01, 0 to 4) at which -log(1 - R) + b log size orders the six; the intercept cancels."""
    ok = [b for b in np.arange(0, 4.001, 0.01) if orders({i: -math.log(1 - r[i][m]) + b * math.log(size[i]) for i in KNOWN})["orders"]]
    return {"b_min": float(min(ok)), "b_max": float(max(ok))} if ok else {"b_min": None, "b_max": None}


def half_band(r: dict[str, dict[str, Any]], m: str) -> float:
    """Largest half-1 minus half-2 difference of the size-removed score over the six (size and fit cancel)."""
    return max(abs(math.log(1 - r[i]["halves"][m][0]) - math.log(1 - r[i]["halves"][m][1])) for i in KNOWN)


def orders(v: dict[str, float | None], higher_is_reuse: bool = True) -> dict[str, Any]:
    """Every built reuse edit above both built new edits (on the measure's reuse-is-higher orientation)."""
    if any(v.get(i) is None for i in KNOWN):
        return {"read": False}
    sgn = 1 if higher_is_reuse else -1
    lo, hi = min(sgn * v[i] for i in REUSE), max(sgn * v[i] for i in NEW)  # type: ignore[operator]
    return {"read": True, "orders": lo > hi, "margin": lo - hi, "lowest_reuse": min(REUSE, key=lambda i: sgn * v[i])}  # type: ignore[operator]


def pair_gaps(pairs: dict[str, Any], get: Any) -> dict[str, float]:
    return {p: mean(get(i) for i in v["reuse"]) - mean(get(i) for i in v["new"]) for p, v in pairs.items()}


def vs_size(g: dict[str, float], pairs: dict[str, Any]) -> dict[str, Any]:
    minus_size = [pairs[p]["smaller_update"] for p in g]
    rho = np.corrcoef(rankdata(list(g.values())), rankdata(minus_size))[0, 1]  # Spearman
    return {"rho_vs_minus_size": float(rho), "reuse_higher": sum(v > 0 for v in g.values()), "n": len(g)}


def gain_reach(pairs: dict[str, Any]) -> dict[str, float]:
    sc = load(E231)["pairs"]
    return {p: mean(t["kl"]["reach"] for t in sc[p[5:]]["reuse"].values()) - mean(t["kl"]["reach"] for t in sc[p[5:]]["new"].values()) for p in pairs}


def main() -> int:
    pairs = {p: v for p, v in load(PAIRS_JSON)["pairs"].items() if p.startswith("e187_")}
    trained = sorted({i for v in pairs.values() for i in (*v["reuse"], *v["new"])})
    items = ["do_nothing", *KNOWN, *trained]
    r = {i: readings(i) for i in items}
    filed = load(VPD / "summary.json")["per_item"]
    dev = max(abs(r[i]["R"] - filed[i]["R_read"]) for i in items if i != "do_nothing")
    assert dev < 1e-6, f"measure 1 does not reproduce E226's R ({dev})"
    size = sizes([*KNOWN, *trained])
    kl = {i: item_json(i)["kl_fm"] for i in [*KNOWN, *trained]}
    band = {m: max(abs(r[i]["halves"][m][0] - r[i]["halves"][m][1]) for i in KNOWN) for m in MEASURES}

    out: dict[str, Any] = {"experiment": "P1.42 r_reanalysis", "do_nothing": {m: r["do_nothing"][m] for m in MEASURES},
                           "max_dev_R_vs_E226": dev, "band": band, "known": {}, "pairs": {}, "fits": {}}  # fmt: skip
    for m in MEASURES:
        v = {i: r[i][m] for i in KNOWN}
        out["known"][m] = {"values": v, **orders(v)}
        g = pair_gaps(pairs, lambda i, m=m: r[i][m])
        out["pairs"][m] = {"gaps": g, **vs_size(g, pairs)}
        z, ab = size_removed(r, size, m)
        _, ab_pairs = size_removed({i: r[i] for i in trained}, {i: size[i] for i in trained}, m)
        out["fits"][m] = {"a_b": ab, "a_b_pairs_only": ab_pairs, "orders_for_slopes": slope_window(r, size, m),
                          "band_size_removed": half_band(r, m)}  # fmt: skip
        out["known"][f"{m}_size_removed"] = {"values": {i: z[i] for i in KNOWN}, **orders({i: z[i] for i in KNOWN})}
        gz = pair_gaps(pairs, lambda i, z=z: z[i])
        out["pairs"][f"{m}_size_removed"] = {"gaps": gz, **vs_size(gz, pairs)}
    out["side"] = {k: {i: r[i]["side"][k] for i in ["do_nothing", *KNOWN]} for k in r["x2"]["side"]}
    out["side_orders"] = {k: orders({i: r[i]["side"][k] for i in KNOWN}) for k in r["x2"]["side"]}

    blk = load(E232)["levels"]["block"]["read"]
    resid = {p: blk["gap_matched"][p] - blk["gap_c2_law"][p] for p in pairs}
    out["pairs"]["blocks_matched"] = {"gaps": blk["gap_matched"], **vs_size(blk["gap_matched"], pairs)}
    out["pairs"]["blocks_minus_c2_rule"] = {"gaps": resid, **vs_size(resid, pairs)}

    out["known"]["update_size"] = {"values": {i: size[i] for i in KNOWN}, **orders({i: size[i] for i in KNOWN}, False)}
    out["known"]["kl_fm"] = {"values": {i: kl[i] for i in KNOWN}, **orders({i: kl[i] for i in KNOWN}, False)}
    out["pairs"]["kl_fm"] = {"gaps": (g := pair_gaps(pairs, lambda i: -kl[i])), **vs_size(g, pairs)}
    s3 = {p: v["old"]["S3_held_model"] for p, v in pairs.items()}
    out["pairs"]["held_S3_saved_size"] = {"gaps": s3, **vs_size(s3, pairs)}
    out["known"]["held_S3"] = {"values": {"rebuilt": load(E197)["S"]["rebuilt"]["3"]}, "read": False}
    out["pairs"]["gain_reach"] = {"gaps": (g := gain_reach(pairs)), **vs_size(g, pairs)}
    out["sizes_known"] = {i: size[i] for i in KNOWN}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    for k in out["known"]:
        print(k, {a: b for a, b in out["known"][k].items() if a != "values"})
    for k in out["pairs"]:
        print(k, {a: b for a, b in out["pairs"][k].items() if a != "gaps"})
    print("band", band, "do_nothing", out["do_nothing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
