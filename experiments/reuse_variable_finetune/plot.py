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


if __name__ == "__main__":
    main()
