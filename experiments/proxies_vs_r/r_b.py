"""Cell P2.51: the proxies scored against E_b, the reuse reading under definition (b), from E257's saved files.

    ~/commute-agents/fleet/cpu-run --mem 2G python3 experiments/proxies_vs_r/r_b.py

Definition (b) (user, 2026-11-21) reads reuse over the parts the fine-tune changed, weighted by their change; E257
(P1.50, definition-lead) read it as E_b with the untouched share beside it. Sources: E257's
rb_states/results/summary_items.json: .built_reference_P1_49 for the built edits (P1.49's E_b; the moved edit is not
in it), .pairs.<pair>.gap_E_b (reuse side minus new side, size-matched, every seed), .keys.<task>.E_b (mean over seeds
0-2 at E237's common size), .randoms (two random updates) and .stage0.floor (bf16 against fp32 on x2).

Writes results/r_b_vpd.json in analyse.py's `--r` form plus E_b's spread against its floor and the untouched shares,
results/table_r_b_vpd.json (analyse.table), and figures/gaps_vs_r_b_and_r_a.png: each proxy's pair gap against E_b's
gap and against R_A's gap (r_a_vpd.json), with each score's random band.
"""

from __future__ import annotations

import json
from typing import Any

import analyse as A
import numpy as np
from r_a import SEEDS

from commute import figures as F

HERE = A.HERE
SRC = "rb_states/results/summary_items.json"


def r_b() -> dict[str, Any]:
    s = A.load(SRC)
    ref = s["built_reference_P1_49"]
    keys = s["keys"]
    assert all(keys[t]["n"] == len(SEEDS) for t in keys), "a key task without all three seeds"
    untouched = {"built": {e: ref[e]["untouched_share"] for e in A.BUILT if e in ref}, "trained_max": s["trained"]["untouched_max"]}
    missing = [e for e in A.BUILT if e not in ref] + [p for p in A.PAIRS if f"e187_{p}" not in s["pairs"]] + [t for t in A.KEY if t not in keys]
    return {
        "built": {e: ref[e]["E_b"] for e in A.BUILT if e in ref},
        "pairs": {p: s["pairs"][p]["gap_E_b"] for p in (f"e187_{q}" for q in A.PAIRS) if p in s["pairs"]},
        "key": {t: keys[t]["E_b"] for t in A.KEY if t in keys},
        "untouched_share": untouched,
        "not_read": missing,
    }


def spread(r: dict[str, Any]) -> dict[str, Any]:
    """E_b's range on each item set beside the random updates' reading (the scale E257 sets a gap against) and the
    bf16/fp32 floor; the largest seed range within one side of a pair is the seed noise a pair gap sits on."""
    s = A.load(SRC)
    rnd = [v["E_b"] for v in s["randoms"].values()]
    sides = [s["pairs"][p][side] for p in r["pairs"] for side in ("reuse", "new")]
    gaps, key, built = list(r["pairs"].values()), list(r["key"].values()), list(r["built"].values())
    return {
        "random": rnd, "random_max": max(rnd), "floor_bf16_fp32": s["stage0"]["floor"],
        "trained_min": s["trained"]["E_b_min"], "trained_max": s["trained"]["E_b_max"], "trained_n": s["trained"]["n_items"],
        "trained_max_over_random": s["trained"]["E_b_max"] / max(rnd), "trained_min_over_random": s["trained"]["E_b_min"] / max(rnd),
        "pair_gap_min": min(gaps), "pair_gap_max": max(gaps), "pairs_abs_gap_over_random_max": max(abs(g) for g in gaps) / max(rnd),
        "pairs_inside_random_band": sum(abs(g) <= max(rnd) for g in gaps), "pairs_n": len(gaps),
        "seed_range_max_within_side": max(d["E_b_max"] - d["E_b_min"] for d in sides),
        "key_min": min(key), "key_max": max(key), "key_range_over_random": (max(key) - min(key)) / max(rnd),
        "built_min": min(built), "built_max": max(built), "built_n": len(built),
    }  # fmt: skip


def plot(rows: dict[str, Any], rb: dict[str, float], ra: dict[str, float], band: dict[str, float]) -> Any:
    fig, axes = F.figure((10.0, 12.0), ncols=4, nrows=5)
    data: list[dict[str, Any]] = []
    for k, p in enumerate(A.PROXIES):
        y = rows[p]["values"]["pairs"]
        for j, (score, x, words) in enumerate((("E_b", rb, "E_b gap, definition (b)"), ("R_A", ra, "R_A gap, wording A"))):
            ax = axes.flat[2 * k + j]
            ids = A.common(y, x)
            ax.axvspan(-band[score], band[score], color="#dddddd", label=f"within ± the random update's {score}")
            ax.axvline(0, color="#999999", linewidth=0.6)
            ax.axhline(0, color="#999999", linewidth=0.6)
            ax.scatter([x[i] for i in ids], [y[i] for i in ids], color="#0072B2" if score == "E_b" else "#D55E00", s=20)
            rho = A.spearman(y, x)["rho"]
            ax.set_title(f"{A.PROXIES[p][0]}\nagainst {score}: Spearman " + ("not defined" if rho is None else f"{rho:+.2f}") + f", n {len(ids)}", fontsize=7)
            ax.set_xlabel(words, fontsize=7)
            ax.set_ylabel("proxy gap, reuse minus new", fontsize=7)
            ax.tick_params(labelsize=6)
            data += [{"proxy": p, "score": score, "pair": i, "score_gap": x[i], "proxy_gap": y[i], "band": band[score]} for i in ids]
    for ax in axes.flat[:2]:
        ax.legend(loc="lower left", fontsize=6, frameon=False)
    fig.suptitle("Does any proxy's pair gap follow the score's? Left of each pair of panels: E_b under definition (b); right: R under "
                 "wording A.\nE_b's gaps sit inside its random band or below it; R_A's all lie above zero", fontsize=9)  # fmt: skip
    return F.save(fig, HERE / "figures" / "gaps_vs_r_b_and_r_a.png", data=data,
                  n="nine built pairs, each gap a reuse-side mean minus a new-side mean over 2 to 8 seeds, size-matched; "
                    "band: ± one random update's reading (E_b 0.011, R_A 0.30), E257's construction for E_b, applied to R_A alike")  # fmt: skip


def main() -> int:
    r = r_b()
    sp = spread(r)
    print("not read:", r["not_read"])
    print({k: v for k, v in sp.items() if k != "random"})
    random = (f"{A.inside(sp['random'], sp['trained_min'], sp['trained_max'])} inside; random {min(sp['random']):.4g} to "
              f"{sp['random_max']:.4g}, trained fine-tunes {sp['trained_min']:.4g} to {sp['trained_max']:.4g} (E257's 2 random updates)")  # fmt: skip
    out = {"name": "r_b_vpd", "featurization": "vpd", "random": random, "spread": sp, **r}
    (HERE / "results" / "r_b_vpd.json").write_text(json.dumps(out, indent=1))
    rows = A.table({s: r[s] for s in A.SETS}, A.proxies(), "vpd")
    rows["R"]["random_inside_finetune_range"] = random
    (HERE / "results" / "table_r_b_vpd.json").write_text(json.dumps({"cell": "P2.51", "R": "r_b_vpd", "rows": rows}, indent=1))
    for p, row in rows.items():
        if "vs_R" in row:
            print(p, {s: (v["rho"] if v["rho"] is None else round(v["rho"], 2), v["n"]) for s, v in row["vs_R"].items()},
                  {c: {s: v["rho"] and round(v["rho"], 2) for s, v in part.items()} for c, part in row["vs_R_partial"].items() if part})  # fmt: skip
    ra = json.loads((HERE / "results" / "r_a_vpd.json").read_text())
    ra_random = float(np.max([v["vpd"] for v in A.load("sae_r_a_random/results/summary.json")["D1"].values()]))
    print(plot(rows, r["pairs"], ra["pairs"], {"E_b": sp["random_max"], "R_A": ra_random}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
