"""R2's one figure, from results/<pair>_s<seed>.json, R1's results/rules.json and predictions.json only (no model). Each fine-tune is a
point at its found G (k = 4, the best site by the highest of the three starts) and its found H (best over the five sites), by the
rule R1's rules.py reads, on the same axes as R1's four pairs (hollow markers) and the filed cut-offs from E137. No labels: the
reading is the lead's, under the filed rule. Reversion and setter mismatch at the best G site are saved in the CSV beside the figure.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/reuse_variable_finetune/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "reuse_variable_test" / "results"
PAIRS = ("Always on", "Cue", "XOR", "Table")
COLORS = dict(zip(PAIRS, ("#0072B2", "#D55E00", "#009E73", "#E69F00"), strict=True))  # Okabe-Ito, as `figures.METHODS`


def best(out: dict[str, Any]) -> dict[str, Any] | None:
    """Found G and H as R1 reads them: the site of the highest read G over its three starts; H the best site. None: no reading."""
    if out.get("no_reading") or not out.get("complete"):
        return None
    sites = sorted({k.split("/")[1] for k in out["G"]})
    by = {s: [v for k, v in out["G"].items() if k.split("/")[1] == s] for s in sites}
    site = max(sites, key=lambda s: max(v["G"] for v in by[s]))
    top = max(by[site], key=lambda v: v["G"])
    h = max(out["H"].values(), key=lambda v: v["H"])
    return {"G": top["G"], "site": site, "reversion": top["reversion"], "mismatch": top["mismatch"], "H": h["H"], "H_reversion": h["reversion"], "H_mismatch": h["mismatch"]}


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    cut = json.loads((results / "predictions.json").read_text())["cutoffs"]
    r1 = json.loads((R1 / "rules.json").read_text())["reading"]["k4"]
    fig, ax = FG.figure(FG.ONE_COLUMN)
    recs: list[dict[str, Any]] = []
    empty: list[str] = []
    for path in sorted(results.glob("*_s[0-2].json")):
        if path.name.startswith("amend"):  # the amendments' files have their own figure
            continue
        rec = json.loads(path.read_text())
        pt = best(rec)
        if pt is None:
            empty.append(f"{rec['item']}: {rec.get('no_reading', 'unfinished')}")
            continue
        ax.scatter(pt["G"], pt["H"], color=COLORS[rec["pair"]], marker="o", s=22, alpha=0.85, linewidths=0, label=rec["pair"] if rec["seed"] == 0 else None)
        recs.append({"item": rec["item"], "pair": rec["pair"], "seed": rec["seed"], **pt, "gap": rec["dist"]["gap"], "tv_to_F": rec["dist"]["tv_to_F"]})
    for pair in PAIRS:
        g, h = r1[pair]["best_G"], r1[pair]["best_H"]
        ax.scatter(g, h, facecolors="none", edgecolors=COLORS[pair], marker="s", s=44, linewidths=1.2, label=f"{pair}, the built pair (R1)")
    ax.axvline(cut["G"], color="#777777", linewidth=0.8, linestyle="--")
    ax.axhline(cut["H"], color="#777777", linewidth=0.8, linestyle="--")
    ax.text(cut["G"], 0.02, " re-weighted cut-off", fontsize=6, color="#555555", transform=ax.get_xaxis_transform(), va="bottom")
    ax.text(0.0, cut["H"], " map-found cut-off", fontsize=6, color="#555555", va="bottom")
    ax.set_xlabel("found G at k = 4 (best site, best of three starts)")
    ax.set_ylabel("found H (best site)")
    ax.set_title("Do fine-tunes imitating a pair land where the built pair landed?", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    note = "dashed: cut-offs filed from E137; " + ("; ".join(empty) if empty else "no item without a reading")
    FG.save(fig, out / "finetunes_vs_pairs.png", n=f"{len(recs)} fine-tunes, d on 100,000 held-out windows per pair; {note}", data=recs)


def _in2(rec: dict[str, Any]) -> dict[str, Any]:
    """The in2 start with the highest read G of a record's G block."""
    return max((v for k, v in rec["G"].items() if k.split("/")[1] == "in2"), key=lambda v: v["G"])


def leftover_figure(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    """Amendments 1 and 2: G at in2 (best of three starts) against 1 - TV(fine-tune, pair) / gap, per seed and checkpoint, with
    E137's witness read against the same fine-tune (hollow) and E139's fine-tunes (grey). Dashed: the re-weighted cut-off;
    dotted: y = x, where G would sit if the leftover alone set it."""
    cut = json.loads((results / "predictions.json").read_text())["cutoffs"]["G"]
    a1 = json.loads((results / "amend1.json").read_text())["items"] if (results / "amend1.json").is_file() else {}
    fig, ax = FG.figure(FG.ONE_COLUMN)
    recs: list[dict[str, Any]] = []

    def point(label: str, step: int, rec: dict[str, Any], wit: dict[str, Any] | None, color: str) -> None:
        d = rec["dist"]
        x, g = 1 - d["tv_to_F"] / d["gap"], _in2(rec)
        ax.scatter(x, g["G"], color=color, s=22, linewidths=0)
        if wit:
            ax.scatter(x, wit["G"], facecolors="none", edgecolors=color, s=30, linewidths=1.0)
        h = max(rec["H"].values(), key=lambda v: v["H"]) if rec.get("H") else {}
        recs.append({"item": label, "step": step, "gap": d["gap"], "tv_to_F": d["tv_to_F"], "one_minus_share": x, "G_in2": g["G"],
                     "G_best": max(v["G"] for v in rec["G"].values()), "reversion_in2": g["reversion"], "mismatch_in2": g["mismatch"],
                     "H_best": h.get("H"), "witness_G": wit["G"] if wit else None, "witness_reversion": wit["reversion"] if wit else None})

    for s in (0, 1, 2):
        rec = json.loads((results / f"always_on_s{s}.json").read_text())
        point(f"E139 always_on_s{s}", 300, rec, a1.get(f"always_on_s{s}"), "#999999")
        p = results / f"amend2_always_on_s{s}.json"
        for step, r in sorted(json.loads(p.read_text())["steps"].items() if p.is_file() else [], key=lambda kv: int(kv[0])):
            if r.get("complete") and not r.get("no_reading"):
                point(f"amend2 always_on_s{s}", int(step), r, r.get("witness"), COLORS["Always on"])
    ax.axhline(cut, color="#777777", linewidth=0.8, linestyle="--")
    ax.text(0.0, cut, " re-weighted cut-off", fontsize=6, color="#555555", transform=ax.get_yaxis_transform(), va="bottom")
    FG.reference(ax, 0.9, "reached")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.8, linestyle=":", label="G = 1 - leftover / gap")
    if a1:  # the hand-made pair read by its own witness, at zero leftover
        pair_g = json.loads((results / "amend1.json").read_text())["control"]["G"]
        ax.scatter(1.0, pair_g, marker="s", facecolors="none", edgecolors="black", s=40, linewidths=1.0, label="the hand-made pair")
    ax.scatter([], [], color="#999999", s=22, linewidths=0, label="300 steps at 3e-5 (E139)")
    ax.scatter([], [], color=COLORS["Always on"], s=22, linewidths=0, label="300 / 1000 / 3000 steps at 1e-5 (amendment 2)")
    ax.scatter([], [], facecolors="none", edgecolors="#555555", s=30, linewidths=1.0, label="hollow: the pair's own read-out")
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.set_xlabel("1 - TV(fine-tune, pair) / gap")
    ax.set_ylabel("G at in2 (best of three starts)")
    ax.set_title("Does G rise as the fine-tune gets closer to the pair?", fontsize=9)
    FG.save(fig, out / "g_vs_leftover.png", n=f"{len(recs)} fine-tune states; filled: found G, hollow: E137 witness; grey: E139; right of the reached line the leftover is at most 0.1 of the gap", data=recs)


if __name__ == "__main__":
    main()
    leftover_figure()
