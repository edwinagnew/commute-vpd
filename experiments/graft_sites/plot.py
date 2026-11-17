"""E206 figures, from results/summary.json and results/stage0.json (no GPU, no model):
    figures/survival.png  left: one row per graft, survival S under the per-site map (reading 3) and one gain per unit
                          (reading 1), with the lowest of E197's old-task fine-tunes and E201's graft as reference lines;
                          a graft with no reading is named as such. Right: reading 3's S of the four last-layer head
                          grafts against their offset's largest M-head share, E201's graft beside them.

    ~/commute-agents/fleet/cpu-run python experiments/graft_sites/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import FULL_WIDTH, figure, reference, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
PRED = json.loads((RESULTS / "predictions.json").read_text())
E201_S3 = 0.13355354448066925  # experiments/graft_negative/results/stage3.json, reading 3
LAYER = {"early": PRED["settings"]["early_layer"], "last": PRED["settings"]["last_layer"]}
LAST = ("last_lowest", "last_rank5", "last_rank9", "last_rank13")


def main() -> None:
    summ, s0 = json.loads((RESULTS / "summary.json").read_text()), json.loads((RESULTS / "stage0.json").read_text())
    k = {"last_lowest": summ["k"]["lowest"], "early_lowest": summ["k"]["lowest"], **{f"last_rank{r}": v for r, v in summ["k"].items() if r != "lowest"}}
    names = {role: ("MLP neurons, last layer" if role == "neurons" else f"head, layer {LAYER['early' if role.startswith('early') else 'last']}, k {k[role]}") for role in summ["S"]}
    fig, (a, b) = figure(FULL_WIDTH, ncols=2)
    rows = []
    for i, (role, S) in enumerate(summ["S"].items()):
        if S["3"] is None:
            a.annotate("no reading", (0.0, i), fontsize=6, color="#555555", va="center")
        else:
            a.scatter([S["3"]], [i], color="#009E73", s=24, label="per-site map (reading 3)" if i == 0 else None)
            a.scatter([S["1"]], [i], color="#0072B2", marker="s", s=18, label="one gain per unit (reading 1)" if i == 0 else None)
        rows.append({"graft": names[role], "S3": S["3"], "S1": S["1"], "S2": S["2"]})
    reference(a, PRED["rules"]["R1"]["old_task_floor"], "lowest old-task fine-tune")
    reference(a, E201_S3, "E201's graft")
    a.set_yticks(range(len(names)), list(names.values()), fontsize=6)
    a.invert_yaxis()
    a.set_xlabel("survival S of the grafted model")
    a.legend(fontsize=6, loc="lower right")
    a.set_title("Survival per graft", fontsize=8)
    share = s0["max_share_by_offset"]
    pts = [(share[str(k[r])], summ["S"][r]["3"]) for r in LAST if summ["S"][r]["3"] is not None]
    b.scatter([p[0] for p in pts], [p[1] for p in pts], color="#009E73", s=24, label="this run's last-layer heads")
    b.scatter([share["16"]], [E201_S3], color="#777777", marker="x", s=24, label="E201's graft (k 16)")
    b.set_xlabel("largest M-head share at the graft's offset")
    b.set_ylabel("reading 3 survival S")
    b.legend(fontsize=6)
    b.set_title("Last-layer heads by offset share", fontsize=8)
    rows += [{"graft": f"share {x:.4f}", "S3": y, "S1": None, "S2": None} for x, y in pts]
    save(fig, FIGURES / "survival.png", n="one build per graft; 512 read windows, last position; fits on 4,096 windows x 32 positions", data=rows)


if __name__ == "__main__":
    main()
