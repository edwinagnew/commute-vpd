"""E107 (label_vs_style) figures from results/*.json and the stored sets of E103 and E105; no model. Colours as E105's
plots: the known-facts fine-tunes purple, the fictional fine-tune orange, references grey; the wrong-answer arm teal.

  figures/overlap.png  carrying-set overlaps on each probe: the known and wrong seeds, E105's cross-half pairs, wrong in
                       known, wrong in fictional and known in fictional, with the filed rule's two lines
  figures/topic.png    the topic split on each probe: each topic's seeds, E105's random halves, authors in world facts,
                       and each topic inside the fictional sets, with the filed "topic moves the set" line
  figures/gain.png     the wrong arm's content gain on its own (wrong) sentences and on the true ones, beside the known
                       arm's gain on the true sentences and C1's bar
  figures/amend1_sets.png  Amendment 1: Jaccard of VPD's switched-on sets on E94's maths rows, each arm's seeds, wrong
                       with known (scored), wrong and known with fictional, and the filed rule's two lines

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/label_vs_style/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
KAP = HERE.parent / "known_answer_pair" / "results"
E105 = HERE.parent / "rows_null" / "results"
E102 = HERE.parent / "qwen_carrying" / "results" / "states"
PURPLE, TEAL, ORANGE, GREY, INK, GRID = "#8a4fbf", "#00a088", "#eb6834", "#7a7a75", "#52514e", "#e4e3df"
SEEDS = (42, 142)
WORDS = {"real": "real rows", "tofu": "TOFU rows (scored)"}


def js(path: Path) -> Any:
    return json.loads(path.read_text())


def set_path(name: str, rows: str) -> Path:
    """Where each arm's matched carrying set lives, as run.py's `stored_set` for the stored ones."""
    if name.startswith(("WRONG", "AUTH", "WORLD")):
        return RES / "sets" / f"{name}__{rows}.json"
    if name[0] in "AB" and name[1:].isdigit():
        return E105 / "sets" / f"{name}__{rows}.json"
    if name == "KNOWN42":
        return KAP / "neurons_amend1" / f"a1_matched__{rows}.json"
    if name == "KNOWN142":
        return KAP / "neurons_amend2" / f"known_s142_matched__{rows}.json"
    s = name.removeprefix("FICT")
    return KAP / "neurons" / f"fict_s{s}_matched__real.json" if rows == "real" else E102 / f"a2_s{s}_matched.json"


def share(a: str, b: str, rows: str) -> float:
    """The share of a's set inside b's, as qwen_neurons.overlap."""
    sa, sb = (set(js(set_path(n, rows))["carry"]) for n in (a, b))
    return len(sa & sb) / len(sa)


def seeds(arm: str, rows: str) -> list[float]:
    return [share(f"{arm}142", f"{arm}42", rows)]


def four(a: str, b: str, rows: str) -> list[float]:
    return [share(f"{a}{s}", f"{b}{t}", rows) for s in SEEDS for t in SEEDS]


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def strip(ax: Any, cats: list[tuple[str, list[float], str, str, bool]]) -> None:
    """One column per comparison: its overlaps as markers, filled or hollow, with the column's range written above."""
    for i, (_, vals, colour, marker, filled) in enumerate(cats):
        xs = [i + d for d in ((0.0,) if len(vals) == 1 else (-0.15, -0.05, 0.05, 0.15))]
        ax.scatter(xs, vals, s=46, marker=marker, color=colour if filled else "white", edgecolor=colour, lw=1.6, zorder=3)
        txt = f"{vals[0]:.2f}" if len(vals) == 1 else f"{min(vals):.2f}-{max(vals):.2f}"
        ax.text(i, max(vals) + 0.03, txt, ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cats)), [c[0] for c in cats], fontsize=6.6)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.axhline(0.006, color=GREY, ls=":", lw=1, label="two random sets, about 0.006")
    style(ax)


def overlap(score: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for ax, rows in zip(axes, ("real", "tofu"), strict=True):
        s = score[rows]["wrong"]
        strip(ax, [("known arm:\nseed 142\nin seed 42", seeds("KNOWN", rows), PURPLE, "o", True),
                   ("wrong arm:\nseed 142\nin seed 42", seeds("WRONG", rows), TEAL, "o", True),
                   ("E105: one half\nof the true rows\nin the other half", four("A", "B", rows), GREY, "s", True),
                   ("wrong arm\nin known arm\n(the scored pairs)", four("WRONG", "KNOWN", rows), TEAL, "D", True),
                   ("wrong arm\nin fictional\narm", four("WRONG", "FICT", rows), TEAL, "^", False),
                   ("known arm\nin fictional\narm", four("KNOWN", "FICT", rows), ORANGE, "^", True)])
        ax.axhline(s["style_needs_min"], color=PURPLE, ls="--", lw=1.1,
                   label=f"'topic or style' needs every scored pair above {s['style_needs_min']:.3f}")
        ax.axhline(s["familiarity_needs_max"], color=ORANGE, ls="--", lw=1.1,
                   label=f"'familiarity' needs every scored pair below {s['familiarity_needs_max']:.3f}")
        ax.set_title(f"carrying sets ranked on the {WORDS[rows]}: verdict '{s['verdict']}'", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", frameon=True, framealpha=0.9)
    axes[0].set_ylabel("share of the first set's 1,024 neurons\ninside the second set")
    fig.suptitle("Training wrong answers about the same real people leaves the carrying set with the true-answer run, "
                 "as far from the fictional run as the true-answer run is", fontsize=10)
    save(fig, "overlap.png")


def topic(score: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for ax, rows in zip(axes, ("real", "tofu"), strict=True):
        t = score[rows]["topic"]
        strip(ax, [("real authors:\nseed 142\nin seed 42", seeds("AUTH", rows), PURPLE, "o", True),
                   ("world facts:\nseed 142\nin seed 42", seeds("WORLD", rows), PURPLE, "o", False),
                   ("E105: random\nhalves, one in\nthe other", four("A", "B", rows), GREY, "s", True),
                   ("authors in\nworld facts\n(the scored pairs)", four("AUTH", "WORLD", rows), PURPLE, "D", True),
                   ("wrong arm\nin known arm,\nfor scale", four("WRONG", "KNOWN", rows), TEAL, "D", True),
                   ("authors in\nfictional arm", four("AUTH", "FICT", rows), ORANGE, "^", True),
                   ("world facts in\nfictional arm", four("WORLD", "FICT", rows), ORANGE, "^", False)])
        ax.axhline(t["line"], color=PURPLE, ls="--", lw=1.1,
                   label=f"'topic moves the set' needs every scored pair below {t['line']:.3f}")
        ax.axhline(t["L"], color=GREY, ls="--", lw=1.1, label=f"E105's lowest random-half pair, {t['L']:.3f}")
        ax.set_title(f"carrying sets ranked on the {WORDS[rows]}: verdict '{t['verdict']}'", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", frameon=True, framealpha=0.9)
    axes[0].set_ylabel("share of the first set's 1,024 neurons\ninside the second set")
    fig.suptitle("Splitting the true rows by topic moves the carrying set further than a random split, and further than "
                 "making every answer wrong", fontsize=10)
    save(fig, "topic.png")


def gain(arms: dict[str, Any], known: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, a in enumerate(("WRONG42", "WRONG142")):
        r = arms[a]
        for dx, v, hatch in ((-0.18, r["matched"]["gain"]["content"], None), (0.18, r["second_gain"]["content"], "//")):
            ax.bar(i + dx, v, width=0.34, color=TEAL if hatch is None else "none", edgecolor=TEAL, hatch=hatch)
            ax.text(i + dx, v + 0.03, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.bar(0, 0, color=TEAL, label="the wrong sentences it trained on")
    ax.bar(0, 0, color="none", edgecolor=TEAL, hatch="//", label="the true sentences, never trained")
    ax.axhline(sum(known) / 2, color=PURPLE, ls="--", lw=1.3,
               label=f"known arm on the true sentences it trained on, {min(known):.2f} and {max(known):.2f}")
    bar = arms["WRONG42"]["matched"]["bar"]
    ax.axhline(bar, color=INK, ls=":", lw=1.3, label=f"C1's bar, {bar:.2f}: half the fictional run's own gain")
    ax.set_xticks([0, 1], ["wrong arm, seed 42\n(matched step 32)", "wrong arm, seed 142\n(matched step 31)"], fontsize=8)
    ax.set_ylabel("content gain at the matched step\n(nats a token; 0 = unchanged model)")
    ax.set_ylim(0, 2.8)
    ax.legend(fontsize=7.2, loc="upper right", frameon=True, framealpha=0.9)
    ax.set_title("Trained on wrong answers, the update raises the true answers almost as much\n"
                 "as training on the true answers did", fontsize=10)
    style(ax)
    save(fig, "gain.png")


def vpd_sets(res: dict[str, Any]) -> None:
    """Amendment 1: Jaccard of VPD's switched-on sets on E94's maths rows, from results/amend1_sets.json."""
    def j(a: str, b: str) -> float:
        p = res["pairs"]
        return (p[f"{a}|{b}"] if f"{a}|{b}" in p else p[f"{b}|{a}"])["jaccard"]

    def four(a: str, b: str) -> list[float]:
        return [j(f"{a}_s{s}_matched", f"{b}_s{t}_matched") for s in SEEDS for t in SEEDS]

    r = res["rule"]
    cross = four("wrong", "known")
    assert [round(v, 12) for v in cross] == [round(v, 12) for v in r["cross"]]  # the drawn pairs are the scored ones
    fig, ax = plt.subplots(figsize=(10, 5.0))
    strip(ax, [("known arm:\nseed 142\nwith seed 42", [j("known_s42_matched", "known_s142_matched")], PURPLE, "o", True),
               ("wrong arm:\nseed 142\nwith seed 42", [j("wrong_s42_matched", "wrong_s142_matched")], TEAL, "o", True),
               ("fictional arm:\nseed 142\nwith seed 42", [j("a2_s42_matched", "a2_s142_matched")], ORANGE, "o", True),
               ("wrong arm\nwith known arm\n(the scored pairs)", cross, TEAL, "D", True),
               ("wrong arm\nwith fictional\narm", four("wrong", "a2"), TEAL, "^", False),
               ("known arm\nwith fictional\narm", four("known", "a2"), ORANGE, "^", True)])
    ax.lines[-1].remove()  # strip's random-set line is the neuron overlap's, not Jaccard's
    ax.set_ylim(0.5, 1.0)
    ax.axhline(r["topic_line"], color=PURPLE, ls="--", lw=1.1,
               label=f"'topic or style' needs every scored pair above {r['topic_line']:.4f}")
    ax.axhline(r["familiarity_line"], color=ORANGE, ls="--", lw=1.1,
               label=f"'familiarity' needs every scored pair at or below {r['familiarity_line']:.4f}")
    ax.annotate(f"lowest scored pair {min(cross):.4f},\n{r['topic_line'] - min(cross):.4f} under the line",
                xy=(3.05, min(cross)), xytext=(3.45, 0.74), fontsize=7, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8})
    ax.set_ylabel("Jaccard of the two switched-on sets\n(shared over either; 1 = the same set)")
    ax.legend(fontsize=7, loc="lower left", frameon=True, framealpha=0.9)
    ax.set_title(f"VPD's switched-on sets on the maths rows: the wrong arm sits with the known arm, about a seed apart,\n"
                 f"and its lowest pair misses the 'topic or style' line by {r['topic_line'] - min(cross):.4f}: "
                 f"verdict '{r['verdict']}'", fontsize=9.5)
    save(fig, "amend1_sets.png")


def main() -> None:
    vpd_sets(js(RES / "amend1_sets.json"))
    score = js(RES / "score.json")
    arms = {a: js(RES / f"arm_{a}.json") for a in ("WRONG42", "WRONG142")}
    known = [js(KAP / f"{n}_train.json")["matched"]["gain"]["content"] for n in ("amend1", "amend2")]
    overlap(score)
    topic(score)
    gain(arms, known)
    for rows in ("real", "tofu"):  # every plotted set against the scored file, so a drawn number is a scored one
        assert [round(v, 10) for v in four("WRONG", "KNOWN", rows)] == [round(v, 10) for v in score[rows]["wrong"]["cross"]]
        assert [round(v, 10) for v in four("AUTH", "WORLD", rows)] == [round(v, 10) for v in score[rows]["topic"]["cross"]]
        print(f"{rows}: wrong in E105 half A {[round(v, 4) for v in four('WRONG', 'A', rows)]}; "
              f"authors in known {[round(v, 4) for v in four('AUTH', 'KNOWN', rows)]}, "
              f"world in known {[round(v, 4) for v in four('WORLD', 'KNOWN', rows)]}")
    for a, r in arms.items():
        print(f"{a}: true-sentence gain / own {r['second_gain']['content'] / r['matched']['gain']['content']:.3f}, "
              f"/ known arm's {r['second_gain']['content'] / (sum(known) / 2):.3f}")


if __name__ == "__main__":
    main()
