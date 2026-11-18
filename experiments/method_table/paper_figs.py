"""P2.36, the paper's part-2 figures 3a and 4, from saved results only (no model, no GPU).

A gap is the side built to reuse minus the side built new, as a difference of seed means; on the four twins it is the
real side minus its shuffled or memorising twin. A gap's range runs from (lowest reuse seed - highest new seed) to
(highest reuse seed - lowest new seed), so a range clear of 0 is the scorecard's "every seed beyond every seed" rule.

- Unscaled share, steering and prior: `results/final.json` (the 67M scorecard; prior is log10 of the original model's
  probability of the trained answer, so its gap is in decades).
- Size-matched share and steering, built pairs only: `experiments/size_matched/results/summary.json` (E187), whose
  matched cell keeps the smaller side's seeds and takes the shrunk larger side's. Prior is read on the original model,
  so shrinking an update leaves it unchanged.
- The pair built so the prior and steering disagree: `experiments/disagreeing_pairs2/results/stage2.json` (E199).

    ~/commute-agents/fleet/cpu-run python experiments/method_table/paper_figs.py
"""

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from matplotlib.axes import Axes

from commute.figures import FULL_WIDTH, METHODS, figure, save

ROOT = Path(__file__).resolve().parents[2]
FINAL = "experiments/method_table/results/final.json"
MATCHED = "experiments/size_matched/results/summary.json"
E199 = "experiments/disagreeing_pairs2/results/stage2.json"
OUT = ROOT / "experiments/method_table/figures"
BLUE, GREY, HIGHLIGHT = METHODS["regating"].color, "#555555", "#D55E00"
E199_LABEL = "nearest-word copy vs cipher,\nbuilt so prior and steering disagree"
Point = dict[str, Any]


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def reading(reuse: list[float], new: list[float], src: str, key: str) -> Point:
    """One gap with its seed range; `key` selects the record holding the two seed lists."""
    return {"gap": fmean(reuse) - fmean(new), "lo": min(reuse) - max(new), "hi": max(reuse) - min(new), "n_reuse": len(reuse),
            "n_new": len(new), "source": src, "key": key}  # fmt: skip


def pairs() -> list[Point]:
    """One record per scorecard pair: name, group, and the unscaled and (where run) size-matched readings."""
    final = {(c["pair"], c["method"]): c for c in load(FINAL)["cells"]}
    matched = {(c["pair"], c["reading"]): c for c in load(MATCHED)["cells"] if c["model"] == "67M"}
    out = []
    for pair in dict.fromkeys(p for p, _ in final):
        rec: Point = {"pair": pair, "name": final[(pair, "share")]["pair_name"], "group": final[(pair, "share")]["group"]}
        for m in ("share", "steering", "prior"):
            c = final[(pair, m)]
            rec[m] = reading(c["reuse"], c["new"], FINAL, f'.cells[] | select(.pair=="{pair}" and .method=="{m}")')
        for m in ("share", "steering"):
            if (c := matched.get((pair, m))) is not None:
                v = c["matched_values"]
                rec[f"{m}_matched"] = reading(v["reuse"], v["new"], MATCHED, f'.cells[] | select(.model=="67M" and .pair=="{pair}" and .reading=="{m}") | .matched_values')
        out.append(rec)
    return out


def e199() -> Point:
    s = load(E199)["inputs"]
    rec: Point = {"pair": "e199_neighbour_prior_cipher", "name": E199_LABEL.replace("\n", " "), "group": "built, prior and steering disagree"}
    for m in ("share", "steering"):
        rec[m] = reading(s[m][0], s[m][1], E199, f".inputs.{m}")
    rec["prior"] = {"gap": s["prior_gap"], "source": E199, "key": ".inputs.prior_gap"}
    return rec


def row(fig: str, panel: str, p: Point, series: str, x: Point | None, y: Point) -> Point:
    """One plotted point for the CSV, with the file and jq key behind each coordinate."""
    out = {"figure": fig, "panel": panel, "pair": p["pair"], "pair_name": p["name"], "group": p["group"], "series": series}
    out |= {"y_share_gap": y["gap"], "y_lo": y.get("lo"), "y_hi": y.get("hi"), "y_source": y["source"], "y_key": y["key"]}
    if x is not None:
        out |= {"x_gap": x["gap"], "x_source": x["source"], "x_key": x["key"]}
    return out


def zero(ax: Axes, vertical: bool) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color="#999999", linewidth=0.8, zorder=0)


def fig3a(recs: list[Point]) -> None:
    built = [r for r in recs if r["group"] == "built"]
    twins = [r for r in recs if r["group"] == "twin"]
    ys = {r["pair"]: i for i, r in enumerate(built)} | {r["pair"]: len(built) + 1 + i for i, r in enumerate(twins)}
    fig, ax = figure((5.0, 4.6))
    data = []
    for r in recs:
        y = ys[r["pair"]]
        color, marker = (BLUE, "o") if r["group"] == "built" else (GREY, "s")
        for key, dy, face, label in (("share", -0.16, color, "unscaled"), ("share_matched", 0.16, "white", "sizes matched")):
            if key not in r:
                continue
            s = r[key]
            ax.plot([s["lo"], s["hi"]], [y + dy] * 2, color=color, linewidth=1.2, solid_capstyle="butt")
            ax.scatter([s["gap"]], [y + dy], s=26, marker=marker, facecolor=face, edgecolor=color, linewidths=1.1, zorder=3,
                       label=label if r is built[0] else None)  # fmt: skip
            data.append(row("3a", "share", r, "unscaled" if key == "share" else "matched", None, s))
    zero(ax, vertical=True)
    ax.annotate("no difference", (0, -0.45), xytext=(3, 0), textcoords="offset points", fontsize=7, color=GREY, va="center")
    ax.axhline(len(built), color="#cccccc", linewidth=0.8)
    ax.text(0.99, len(built) + 0.15, "twins: real minus shuffled or memorising twin", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=GREY)
    ax.set_yticks(list(ys.values()), [r["name"] for r in recs])
    ax.set_ylim(len(ys) + 0.8, -0.7)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("share gap, reuse side minus new side")
    ax.set_title("Share gap per 67M pair,\nunscaled and with update sizes matched", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.01, 0.62), framealpha=1, edgecolor="none")
    n = ("Point: difference of seed means; bar: every reuse seed minus every new seed. 2 to 8 seeds a side, as on the "
         "scorecard. Size matching was run on the nine built pairs only.")
    save(fig, OUT / "paper_fig3a_share.png", n=n, data=data)


def panel(ax: Axes, recs: list[Point], extra: Point, x: str, name: str) -> list[Point]:
    data = []
    for r in recs:
        color, marker = (BLUE, "o") if r["group"] == "built" else (GREY, "s")
        sx, sy = r[x], r["share"]
        ax.scatter([sx["gap"]], [sy["gap"]], s=28, marker=marker, color=color, zorder=3)
        data.append(row("4", name, r, "unscaled", sx, sy))
        if "share_matched" in r and (mx := r.get(f"{x}_matched", r[x] if x == "prior" else None)) is not None:
            my = r["share_matched"]
            ax.plot([sx["gap"], mx["gap"]], [sy["gap"], my["gap"]], color=color, linewidth=0.7, zorder=2)
            ax.scatter([mx["gap"]], [my["gap"]], s=28, marker=marker, facecolor="white", edgecolor=color, linewidths=1.0, zorder=3)
            data.append(row("4", name, r, "matched", mx, my))
    ax.scatter([extra[x]["gap"]], [extra["share"]["gap"]], s=90, marker="*", color=HIGHLIGHT, zorder=4)
    data.append(row("4", name, extra, "unscaled", extra[x], extra["share"]))
    zero(ax, vertical=True)
    zero(ax, vertical=False)
    return data


def fig4(recs: list[Point], extra: Point) -> None:
    fig, (ap, asteer) = figure((FULL_WIDTH[0], 3.6), ncols=2)
    data = panel(ap, recs, extra, "prior", "prior") + panel(asteer, recs, extra, "steering", "steering")
    asteer.sharey(ap)
    ap.set_xlabel("prior gap, decades (log10, reuse minus new)")
    asteer.set_xlabel("steering gap at strength 1 (reuse minus new)")
    ap.set_ylabel("share gap (reuse minus new)")
    ap.set_title("Against the original model's prior", fontsize=9)
    asteer.set_title("Against steering", fontsize=9)
    for ax, x, ha, va, dx, dy in ((ap, "prior", "left", "bottom", 8, 8), (asteer, "steering", "right", "top", -8, -14)):
        ax.annotate(E199_LABEL, (extra[x]["gap"], extra["share"]["gap"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=HIGHLIGHT, va=va, ha=ha)  # fmt: skip
    kw: dict[str, Any] = {"s": 28, "linewidths": 1.0}
    ap.scatter([], [], marker="o", color=BLUE, label="built pair, unscaled", **kw)
    ap.scatter([], [], marker="o", facecolor="white", edgecolor=BLUE, label="same pair, sizes matched", **kw)
    ap.scatter([], [], marker="s", color=GREY, label="real-vs-twin pair", **kw)
    ap.scatter([], [], marker="*", color=HIGHLIGHT, label="pair built to split prior and steering", s=90)
    ap.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Share gap per 67M pair against the prior gap and the steering gap", fontsize=10)
    n = ("One point per pair, difference of seed means; twins read real minus twin. Shrinking an update leaves the prior "
         "unchanged, so a size-matched point moves only vertically on the left. Seed ranges are in figure 3a.")
    save(fig, OUT / "paper_fig4_share_vs_prior_steering.png", n=n, data=data)


def main() -> None:
    recs = pairs()
    fig3a(recs)
    fig4(recs, e199())


if __name__ == "__main__":
    main()
