"""E206 figures, from results/summary.json, results/stage0.json and results/grafts/*_read.json (no GPU, no model):
    figures/survival.png  one row per graft: survival S under the per-site map (reading 3) and one gain per unit
                          (reading 1), against E201's graft, E197's rebuilt circuit, its memorised fact (mean of four
                          seeds) and the lowest of its old-task fine-tunes; a graft with no reading is named as such.
    figures/offset.png    reading 3's S of the four last-layer head grafts against their offset's largest M-head share,
                          E201's graft beside them.
    figures/unfitted.png  reading 3's S of each graft read against the share of the graft's own write that reading 3's
                          per-site map leaves unfitted, E201's graft beside them.

    ~/commute-agents/fleet/cpu-run python experiments/graft_sites/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes

from commute.figures import ONE_COLUMN, figure, save

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
EXPERIMENTS = HERE.parent
PRED = json.loads((RESULTS / "predictions.json").read_text())
E201 = json.loads((EXPERIMENTS / "graft_negative/results/stage3.json").read_text())
E201_S3, E201_UNFIT = E201["graft"]["readings"]["3"]["S"], E201["graft_unfitted_share"]
E197 = json.loads((EXPERIMENTS / "held_model_retention/results/summary.json").read_text())["S"]
FACT_S3 = sum(v["3"] for key, v in E197.items() if key.startswith("fact_")) / 4
REBUILT_S3 = E197["rebuilt"]["3"]
LAYER = {"early": PRED["settings"]["early_layer"], "last": PRED["settings"]["last_layer"]}
LAST = ("last_lowest", "last_rank5", "last_rank9", "last_rank13")
GREEN, BLUE, GREY = "#009E73", "#0072B2", "#777777"
N = "one build per graft; 512 read windows, last position; fits on 4,096 windows x 32 positions"


def line(ax: Axes, x: float, label: str, y: float) -> None:
    """A dashed reference line with its label beside it at height y (axes fraction), so close lines stay readable."""
    ax.axvline(x, color=GREY, linewidth=0.8, linestyle="--")
    ax.annotate(label, (x, y), xycoords=("data", "axes fraction"), xytext=(2, 0), textcoords="offset points",
                fontsize=6, color="#555555", rotation=90, va="bottom")


def names(summ: dict[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    k = {"last_lowest": summ["k"]["lowest"], "early_lowest": summ["k"]["lowest"],
         **{f"last_rank{r}": v for r, v in summ["k"].items() if r != "lowest"}}
    layer = {role: LAYER["early" if role.startswith("early") else "last"] for role in summ["S"]}
    return k, {role: ("256 MLP neurons, layer 3" if role == "neurons" else f"head, layer {layer[role]}, k {k[role]}")
               for role in summ["S"]}


def survival(summ: dict[str, Any], label: dict[str, str]) -> None:
    fig, ax = figure((6.0, 3.6))
    rows = []
    for i, (role, S) in enumerate(summ["S"].items()):
        if S["3"] is None:
            ax.annotate("no reading: failed its mechanism check", (0.3, i), fontsize=6, color="#555555", va="center")
        else:
            ax.scatter([S["3"]], [i], color=GREEN, s=26, label="per-site map (reading 3)" if i == 0 else None, zorder=3)
            ax.scatter([S["1"]], [i], color=BLUE, marker="s", s=18, label="one gain per unit (reading 1)" if i == 0 else None, zorder=3)
        rows.append({"graft": label[role], "S3": S["3"], "S1": S["1"], "S2": S["2"]})
    for x, text in ((E201_S3, "E201's graft"), (REBUILT_S3, "rebuilt circuit, E197"), (FACT_S3, "memorised fact, E197"),
                    (PRED["rules"]["R1"]["old_task_floor"], "lowest old-task fine-tune")):
        line(ax, x, text, 0.02)
        rows.append({"graft": f"reference: {text}", "S3": x, "S1": None, "S2": None})
    ax.set_yticks(range(len(label)), list(label.values()), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("survival S: share of the graft's change the held model keeps")
    ax.legend(fontsize=6, loc="upper center")
    ax.set_title("The MLP-neuron graft survives the per-site map nearly whole; the heads do not", fontsize=8)
    save(fig, FIGURES / "survival.png", n=N, data=rows)


def offset(summ: dict[str, Any], s0: dict[str, Any], k: dict[str, int]) -> None:
    fig, ax = figure(ONE_COLUMN)
    share = s0["max_share_by_offset"]
    pts = [(share[str(k[r])], summ["S"][r]["3"], k[r]) for r in LAST]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=GREEN, s=26, label="this run's last-layer heads")
    for x, y, kk in pts:
        ax.annotate(f"k {kk}", (x, y), xytext=(4, -3), textcoords="offset points", fontsize=6)
    ax.scatter([share["16"]], [E201_S3], color=GREY, marker="x", s=26, label="E201's graft, k 16")
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("largest share of attention any M head puts at k")
    ax.set_ylabel("survival S, per-site map")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("Head grafts keep a little more where M's heads already look", fontsize=8)
    save(fig, FIGURES / "offset.png", n=N, data=[{"k": kk, "max_share": x, "S3": y} for x, y, kk in pts]
         + [{"k": "E201 16", "max_share": share["16"], "S3": E201_S3}])


def unfitted(summ: dict[str, Any], label: dict[str, str]) -> None:
    fig, ax = figure(ONE_COLUMN)
    rows = []
    for role, S in summ["S"].items():
        if S["3"] is None:
            continue
        u = json.loads((RESULTS / "grafts" / f"{role}_read.json").read_text())["graft_unfitted_share"]
        ax.scatter([u], [S["3"]], color=GREEN, s=26, label="this run's grafts" if not rows else None)
        rows.append({"graft": label[role], "unfitted_share": u, "S3": S["3"]})
    for r in rows:
        if r["graft"].startswith("256") or r["graft"].endswith("k 4"):
            ax.annotate(r["graft"], (r["unfitted_share"], r["S3"]), xytext=(4, -3), textcoords="offset points", fontsize=6)
    ax.scatter([E201_UNFIT], [E201_S3], color=GREY, marker="x", s=26, label="E201's graft")
    rows.append({"graft": "E201's graft", "unfitted_share": E201_UNFIT, "S3": E201_S3})
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("share of the graft's own write the per-site map leaves unfitted")
    ax.set_ylabel("survival S, per-site map")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("The map fits nine tenths of the neurons' write, half of a head's", fontsize=8)
    save(fig, FIGURES / "unfitted.png", n=N + "; unfitted share on the fit windows", data=rows)


def main() -> None:
    summ, s0 = json.loads((RESULTS / "summary.json").read_text()), json.loads((RESULTS / "stage0.json").read_text())
    k, label = names(summ)
    survival(summ, label)
    offset(summ, s0, k)
    unfitted(summ, label)


if __name__ == "__main__":
    main()
