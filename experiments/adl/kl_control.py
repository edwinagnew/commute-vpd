"""P2.60 (E266): do the activation difference lens's pair gaps survive regressing each state's D on output KL?

$0, saved results only. E265's lens score D per state of the nine built pairs, regressed by OLS (intercept, pooled over
the unique trained states) on KL, log KL, update norm and KL+norm, with E262's design and per-state KL and norm (E257's
states, the same updates the lens read: both multiplied the larger side by E187's c). Each pair's residual gap is set
beside E_b's residual gap from E262. Rules: results/kl/predictions.json. Writes results/kl/summary.json and the figure.

    ~/commute-agents/fleet/cpu-run python3 experiments/adl/kl_control.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from commute import figures as F
from commute.jobs import EXPERIMENTS, load_experiment, require_filed

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kl"
ITEMS = HERE / "results" / "amend1" / "items"
KLC = load_experiment("kl_control", "run")  # E262's design() and P2.51's spearman (KLC.AN)
E262 = EXPERIMENTS / "kl_control" / "results" / "summary.json"


def read(p: Path) -> Any:
    return json.loads(p.read_text())


def e257_name(it: dict[str, Any]) -> str:
    return it["state"] if it["c"] == 1 else f"{it['state'].replace('/', '-')}_scaled_{it['pair']}"


def lens_items() -> tuple[list[dict[str, Any]], float]:
    items = [read(f) for f in sorted(ITEMS.glob("*.json"))]
    pairs = [it for it in items if it["pair"]]
    floor = max(abs(it["D"]) for it in pairs if it["random"])
    return [it for it in pairs if not it["random"]], floor


def fit(rows: dict[str, dict[str, float]], y: np.ndarray, which: str) -> tuple[dict[str, float], float]:
    x = np.array([[1.0, r["D"]] for r in rows.values()]) if which == "D" else KLC.design(rows, which)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    e = y - x @ beta
    return dict(zip(rows, e.tolist(), strict=True)), float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))


def main() -> None:
    require_filed(OUT / "predictions.json")
    st = read(OUT / "predictions.json")["settings"]
    b = read(E262)["B_pairs"]
    items, floor = lens_items()
    rows: dict[str, dict[str, float]] = {}
    for it in items:
        k = e257_name(it)
        s = b["states"][k]
        assert abs(it["size"] / s["norm"] - 1) <= st["size_rel"], f"{it['name']}: lens size {it['size']} vs E257 {s['norm']}"
        rows[k] = {"D": it["D"], "kl": s["kl"], "logkl": s["logkl"], "norm": s["norm"]}
    sides = {p: {sd: sorted({e257_name(i) for i in items if i["pair"] == p and i["side"] == sd}) for sd in ("reuse", "new")}
             for p in sorted({i["pair"] for i in items})}  # fmt: skip
    gap = lambda d, p: fmean(d[i] for i in sides[p]["reuse"]) - fmean(d[i] for i in sides[p]["new"])
    y = np.array([r["D"] for r in rows.values()])
    eb = {p: b["pairs"][f"e187_{p}"] for p in sides}
    calls = {p: v["adl_call"] for p, v in read(HERE / "results" / "amend1" / "compare.json")["pairs"].items()}
    raw = {p: gap({k: r["D"] for k, r in rows.items()}, p) for p in sides}
    fits: dict[str, Any] = {}
    for which in ["D", *st["regressors"]]:
        resid, r2 = fit(rows, y, which)
        g = {p: gap(resid, p) for p in sides}
        out: dict[str, Any] = {"r2": r2, "gaps": g, "outside_floor": sum(abs(v) > floor for v in g.values()),
                               "apart_inside_floor": sum(abs(g[p]) <= floor for p in sides if calls[p] == "apart"),
                               "large_outside_floor": sum(abs(g[p]) > floor for p in st["large_pairs"])}  # fmt: skip
        if which != "D":
            ebr = {p: eb[p][f"resid_{which}"] for p in sides}
            out |= {"sign_vs_eb_resid": sum((g[p] > 0) == (ebr[p] > 0) for p in sides),
                    "sign_vs_eb_raw": sum((g[p] > 0) == (eb[p]["raw"] > 0) for p in sides),
                    "spearman_vs_eb_resid": KLC.AN.spearman(g, ebr)["rho"], "spearman_vs_eb_raw": KLC.AN.spearman(g, {p: eb[p]["raw"] for p in sides})["rho"]}  # fmt: skip
        fits[which] = out
    k = fits[st["filed_regressor"]]
    rival_a = k["apart_inside_floor"] >= 4 and k["sign_vs_eb_resid"] <= 6 and k["spearman_vs_eb_resid"] < 0.5
    rival_b = k["large_outside_floor"] == len(st["large_pairs"]) and k["spearman_vs_eb_resid"] >= 0.5
    summary = {"experiment": "E266 (adl kl_control)", "cell": "P2.60", "descriptive": True, "floor": floor,
               "n_states": len(rows), "calls": calls,
               "raw": {"gaps": raw, "outside_floor": sum(abs(v) > floor for v in raw.values()),
                       "sign_vs_eb_raw": sum((raw[p] > 0) == (eb[p]["raw"] > 0) for p in sides),
                       "spearman_vs_eb_raw": KLC.AN.spearman(raw, {p: eb[p]["raw"] for p in sides})["rho"]},
               "spearman_D_kl": KLC.AN.spearman({i: r["D"] for i, r in rows.items()}, {i: r["kl"] for i, r in rows.items()})["rho"],
               "control_D_on_D_max_gap": max(abs(v) for v in fits["D"]["gaps"].values()),
               "raw_vs_e265_max_diff": max(abs(raw[p] - v["adl"]) for p, v in read(HERE / "results" / "amend1" / "compare.json")["pairs"].items()),
               "fits": fits, "reading": "rival_a" if rival_a else "rival_b" if rival_b else "mixed", "states": rows}  # fmt: skip
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "states"}, indent=1))
    plot(raw, fits[st["filed_regressor"]]["gaps"], floor)


def plot(raw: dict[str, float], resid: dict[str, float], floor: float) -> None:
    fig, ax = F.figure((5.0, 3.4))
    ax.axvspan(-floor, floor, color="#DDDDDD", label=f"random-update floor, |D| {floor:.4f}")
    ax.axvline(0, color="#777777", linewidth=0.8)
    order = sorted(raw, key=lambda p: raw[p])
    ax.scatter([raw[p] for p in order], range(len(order)), color="#0072B2", marker="o", s=24, label="raw")
    ax.scatter([resid[p] for p in order], range(len(order)), color="#D55E00", marker="s", s=24, label="after regressing D on KL")
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("lens pair gap D, reuse minus new")
    ax.set_title("Lens pair gaps before and after removing output KL", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    F.save(fig, HERE / "figures" / "adl_kl_gaps.png", n="9 built pairs, 2-8 seeds per side, OLS over 76 states",
           data=[{"pair": p, "raw_gap": raw[p], "resid_kl_gap": resid[p], "floor": floor} for p in order])


if __name__ == "__main__":
    main()
