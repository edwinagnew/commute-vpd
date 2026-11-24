"""E288's score (no GPU): the filed rules of results/predictions.json (gate, compared, no_call, pairs_card) on the item
files, beside the definition's reading R_A of the same state at the same scale (E248's items), each item's labelled
share of D, and stage 0's controls -> results/score.json and figures/pairs.png.

    ~/commute-agents/fleet/cpu-run python3 experiments/cmap/score.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RESULTS, REPO = HERE / "results", HERE.parents[1]
P = json.loads((RESULTS / "predictions.json").read_text())
S = P["settings"]
E248 = REPO / "experiments" / "kstar_vpd_wording_a" / "results" / "items"
BAR = 0.07  # the card's smallest_gap: the pairs_card rule's 0.07
REUSE, NEW, RAND = FG.Style("side built to reuse", "#0072B2", "o"), FG.Style("side built new", "#E69F00", "D"), FG.METHODS["random"]
PAIR_NAMES = {  # reuse side / new side, as E187 labels them
    "e46_copy_random": "e46 copy / random map", "e53_lowercase_cipher": "e53 lowercase / cipher (text)",
    "e54_matched": "e54 mid / cipher", "e54_own": "e54 mid / own cipher", "e58_copy_prior_cipher": "e58 copy / prior cipher",
    "e58_rank1_96": "e58 top / cipher", "e62_copy_lookup": "e62 copy / lookup (cued)", "open_pair_c3": "c3 top / cipher",
    "open_pair_c3_copy": "c3 copy / cipher",
}
KEY_NAMES = {"copy": "copy", "rank1": "top", "elicit": "mid", "cipher": "cipher", "prior_cipher": "prior-matched cipher"}


def definition_reading(it: dict[str, Any], pair: str | None) -> float:
    """R_A (E248, main search) of the same state at the same scale; for a key item E248's `_scaled` file."""
    base = it["state"].replace("/", "-")
    names = [f"{base}_scaled.json"] if pair is None else [f"{base}_scaled_{pair}.json", f"{base}.json"]
    path = next(E248 / n for n in names if (E248 / n).is_file())
    rec = json.loads(path.read_text())
    assert math.isclose(rec.get("c") or 1.0, it["c"], abs_tol=1e-3), f"{it['item']}: E248 {path.name} at c {rec.get('c')}"
    assert rec["n_windows"] == it["task_rows"]["D_n"], f"{it['item']}: |D| {it['task_rows']['D_n']} vs E248 {rec['n_windows']}"
    return rec["searches"]["main"]["R_A"]


def row(it: dict[str, Any], pair: str | None) -> dict[str, Any]:
    """One item: the gate, the compared and A+B shares (null under the gate or at gain 0), the labelled share of D."""
    tr, g = it["task_rows"], it["gain"]
    readable = bool(it["clears_gate"]) and g["logodds"] != 0
    rnd = [v["share"]["logodds"] for k, v in it["cmap"].items() if k.startswith("random")]
    return {
        "item": it["item"], "format": tr["format"], "c": it["c"],
        "D_labelled": tr["D_labelled"], "D_n": tr["D_n"], "D_labelled_over_D_n": tr["D_labelled"] / tr["D_n"],
        "rows_used": tr.get("D_used", "D's windows with a defined label"),
        "gain_count": g["count"], "gain_logodds": g["logodds"], "N": it["gate"]["N"], "L": it["gate"]["L"],
        "clears_gate": it["clears_gate"], "circuit_heads": len(it["circuit"]),
        "whole": it["compared"]["whole_logodds_share"] if readable else None,
        "AB": it["compared"]["AB_logodds_share"] if readable else None,
        "random_mean": fmean(rnd) if rnd else None, "random_max": max(rnd) if rnd else None,  # an empty circuit has no random sets
        "R_A": definition_reading(it, pair),
    }


def pairs_card(d: dict[str, float]) -> dict[str, Any]:
    """The filed rule: no call under 5 readable pairs; A: d > 0.07 on ceil(2/3 n) pairs and median d > 0.07; B: median
    |d| <= 0.07; neither: no call. Ties at 0.07 count against A."""
    n = len(d)
    if n < 5:
        return {"n_readable": n, "verdict": "no call"}
    above, need = sum(v > BAR for v in d.values()), math.ceil(2 * n / 3)
    med, med_abs = median(d.values()), median(abs(v) for v in d.values())
    a, b = above >= need and med > BAR, med_abs <= BAR
    verdict = "rival A" if a and not b else "rival B" if b and not a else "no call"
    return {"n_readable": n, "n_above_bar": above, "need": need, "median_d": med, "median_abs_d": med_abs,
            "A_holds": a, "B_holds": b, "verdict": verdict}


def mean_of(xs: list[float | None]) -> float | None:
    """The mean over the values present; None when none is (a side whose circuits are all empty has no random sets)."""
    vs = [x for x in xs if x is not None]
    return fmean(vs) if vs else None


def diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def score_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for pair in S["pairs"]:
        sides = {s: [row(it, pair) for it in items if f"{pair}:{s}" in it["roles"]] for s in ("reuse", "new")}
        readable = all(r["whole"] is not None for rs in sides.values() for r in rs) and all(sides.values())
        mean = {s: {k: mean_of([r[k] for r in rs]) for k in ("whole", "AB", "random_mean", "R_A")} for s, rs in sides.items()}
        out[pair] = {
            "format": sides["reuse"][0]["format"], "readable": readable, "sides": sides, "mean": mean,
            **{f"d_{k}": diff(mean["reuse"][k], mean["new"][k]) if readable else None for k in ("whole", "AB", "random_mean")},
            "definition_gap_R_A": diff(mean["reuse"]["R_A"], mean["new"]["R_A"]),
        }
    return out


def size_corrected(pairs: dict[str, Any]) -> dict[str, Any]:
    """Ours, post hoc, not filed (redteam 2026-11-23): per pair, over items whose circuit is not empty, the reuse-minus-new
    difference of the whole-circuit share, and of the A+B share minus the mean of the item's 10 same-size random head
    sets; and each side's mean circuit size over all its items."""
    out = {}
    for name, p in pairs.items():
        kept = {s: [r for r in rs if r["circuit_heads"] > 0] for s, rs in p["sides"].items()}
        mean = {s: {"whole": mean_of([r["whole"] for r in rs]), "net": mean_of([r["AB"] - r["random_mean"] for r in rs])} for s, rs in kept.items()}
        out[name] = {
            "d_whole_nonempty": diff(mean["reuse"]["whole"], mean["new"]["whole"]),
            "d_AB_minus_random": diff(mean["reuse"]["net"], mean["new"]["net"]),
            "heads": {s: fmean(r["circuit_heads"] for r in rs) for s, rs in p["sides"].items()},
            "n_nonempty": {s: len(rs) for s, rs in kept.items()},
        }
    return out


def controls() -> dict[str, Any]:
    s0 = json.loads((RESULTS / "stage0.json").read_text())
    pv, po, gate = s0["plant_v"], s0["plant_o"], s0["gates"]["e58/copy@1.0000"]
    step = str(pv["step"])
    return {
        "stage0_done": s0["done"], "misses": s0["misses"],
        "hook_floor": {f: v["tol"] for f, v in s0["hook_floor"].items()},
        "plant_v": {"head": pv["head"], "first_clearing_step": pv["step"], "gain_count": pv["gains"][step]["count"],
                    "gain_logodds": pv["gains"][step]["logodds"], "N": gate["N"], "L": gate["L"], "gain_floor": S["gain_floor"],
                    "gains_by_step": pv["gains"], "groups_pruned": pv["groups_pruned"], "cmap_A_share": pv["cmap_A"]},
        "plant_o": {"first_clearing_step": po["step"], "gain_logodds": po["gains"][str(po["step"])]["logodds"],
                    "cmap_share_h": po["cmap"]["h"]["share"]["logodds"], "cmap_share_all": po["cmap"]["all"]["share"]["logodds"]},
        "two_layer": s0["two_layer"]["checks"],
    }


def plot(pairs: dict[str, Any], key: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = FG.figure((7.5, 6.0), ncols=2)
    labels = [*(PAIR_NAMES[p] for p in pairs), *(f"key: {KEY_NAMES[t]}" for t in key)]
    data = []
    for ax, (reading, xlabel) in zip(axes, (("whole", "head CMAP log-odds share on D"), ("R_A", "definition's reading R_A (E248)")), strict=True):
        for i, (name, p) in enumerate(pairs.items()):
            for s, st, off in (("reuse", REUSE, -0.18), ("new", NEW, 0.18)):
                xs = [r[reading] for r in p["sides"][s]]
                ax.scatter(xs, [i + off] * len(xs), color=st.color, marker=st.marker, s=18, alpha=0.85, linewidths=0,
                           label=st.label if i == 0 and reading == "whole" else None)
                if reading == "whole":
                    rx = [r["random_mean"] for r in p["sides"][s] if r["random_mean"] is not None]
                    ax.scatter(rx, [i + off] * len(rx), color=RAND.color, marker=RAND.marker, s=12, alpha=0.5, linewidths=0,
                               label="10 random head sets, mean" if i == 0 and s == "reuse" else None)
                data += [{"panel": reading, "row": name, "side": s, "item": r["item"], "value": r[reading], "random_mean": r["random_mean"]} for r in p["sides"][s]]
        for j, (task, rs) in enumerate(key.items()):
            y = len(pairs) + j
            ax.scatter([r[reading] for r in rs], [y] * len(rs), color="#555555", marker="s", s=18, linewidths=0,
                       label="key task (no built label)" if j == 0 and reading == "whole" else None)
            if reading == "whole":
                rx = [r["random_mean"] for r in rs if r["random_mean"] is not None]
                ax.scatter(rx, [y] * len(rx), color=RAND.color, marker=RAND.marker, s=12, alpha=0.5, linewidths=0)
            data += [{"panel": reading, "row": f"key {task}", "side": "key", "item": r["item"], "value": r[reading], "random_mean": r["random_mean"]} for r in rs]
        ax.axhline(len(pairs) - 0.5, color="#bbbbbb", linewidth=0.8)
        ax.set_yticks(range(len(labels)), labels if reading == "whole" else [""] * len(labels))
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlabel(xlabel)
    FG.reference(axes[0], 1.0, "planted head")
    FG.reference(axes[1], 1.0, "do-nothing")
    fig.legend(loc="outside upper center", ncol=4, fontsize=7, frameon=False)
    axes[0].set_title("Head CMAP: reuse over new by > 0.07 on 5 of 9", fontsize=8, loc="left", pad=12)
    axes[1].set_title("Definition: reuse over new on 9 of 9", fontsize=8, loc="left", pad=12)
    FG.save(fig, HERE / "figures" / "pairs.png", n="one point per fine-tune (2-8 seeds per side, 3 per key task); D = 512 windows, "
            "CMAP reads the 204-500 with a label; E53 on its scorer's held-out windows", data=data)


def main() -> None:
    items = [json.loads(p.read_text()) for p in sorted((RESULTS / "items").glob("*_x*.json"))]
    pairs = score_pairs(items)
    key = {t: [row(it, None) for it in items if "key" in it["roles"] and it["task"] == f"e58/{t}"] for t in S["key_tasks"]}
    allrows = [row(it, None if "key" in it["roles"] and len(it["roles"]) == 1 else it["roles"][-1].split(":")[0]) for it in items]
    d = {p: v["d_whole"] for p, v in pairs.items() if v["readable"]}
    out = {
        "experiment": "E288", "n_items": len(items),
        "gate": {"n_clear": sum(r["clears_gate"] for r in allrows), "n": len(allrows),
                 "not_clear": [r["item"] for r in allrows if not r["clears_gate"]]},
        "no_call_null_shares": [r["item"] for r in allrows if r["whole"] is None],
        "labelled": {"min_ratio": min(r["D_labelled_over_D_n"] for r in allrows),
                     "under_quarter": [r["item"] for r in allrows if r["D_labelled_over_D_n"] < 0.25],
                     "per_task": {it["task"]: f"{it['task_rows']['D_labelled']}/{it['task_rows']['D_n']}" for it in items}},
        "pairs_card": pairs_card(d),
        "pairs_card_without_e53": pairs_card({p: v for p, v in d.items() if p != "e53_lowercase_cipher"}),
        "d_whole": d,
        "d_AB": {p: v["d_AB"] for p, v in pairs.items()},
        "d_random": {p: v["d_random_mean"] for p, v in pairs.items()},
        "definition_gap_R_A": {p: v["definition_gap_R_A"] for p, v in pairs.items()},
        "sign_agrees_with_definition": {p: (v["d_whole"] > 0) == (v["definition_gap_R_A"] > 0) for p, v in pairs.items() if v["readable"]},
        "post_hoc_size_corrected": (sc := size_corrected(pairs)),
        "post_hoc_n_above_bar": sum(v["d_AB_minus_random"] is not None and v["d_AB_minus_random"] > BAR for v in sc.values()),
        "pairs": pairs,
        "key_tasks": {t: {"whole": [r["whole"] for r in rs], "mean_whole": mean_of([r["whole"] for r in rs]),
                          "R_A": [r["R_A"] for r in rs], "mean_R_A": mean_of([r["R_A"] for r in rs]),
                          "random_mean": mean_of([r["random_mean"] for r in rs]), "rows": rs} for t, rs in key.items()},
        "controls": controls(),
    }
    (RESULTS / "score.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("gate", "labelled", "pairs_card", "pairs_card_without_e53", "d_whole", "post_hoc_size_corrected", "post_hoc_n_above_bar")}, indent=1))
    plot(pairs, key)


if __name__ == "__main__":
    main()
