"""E88's figures. Reads result files only, loads no model.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_anchor_checks/plot.py

  figures/count_and_recall.png  the hand count by reader and standard, beside new-word recall at both seeds
  figures/paired_rows.png       the same 100 rephrased questions: which model the third reader credits, two standards
  figures/suffix.png            new-word recall with and without the prompt suffix, unchanged model and update
  figures/known_facts.png       real facts: unchanged model, this update, E81's update, against the 10-point limit
  figures/where_harm.png        real facts by question type: unchanged model and both seeds, against Amendment 1's filed bars
  figures/template_seeds.png    the template fine-tune's two seeds against the unchanged model and the filed minimums
"""
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# E84's colours for the same series: rephrased blue, trained yellow. Validated with validate_palette (light, with aqua).
PARA, TRAINED, BOTH = "#2a78d6", "#eda100", "#1baf7a"
SECOND = "#8b8a85"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10.5, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / f"{name}.json").read_text())


def dot(ax: Any, x: float, y: float, color: str, marker: str = "o", label: str | None = None) -> None:
    ax.plot(x, y, marker, color=color, ms=9, mec="white", mew=1.4, zorder=3, label=label)


def with_interval(ax: Any, x: float, arm: dict[str, Any], color: str, marker: str, label: str | None) -> None:
    lo, hi = arm["ci"]
    ax.plot([x, x], [lo, hi], color=color, lw=2, zorder=2, solid_capstyle="round")
    dot(ax, x, arm["share"], color, marker, label)


def hand_panel(ax: Any) -> None:
    filed, third = load("hand_score"), load("hand_third_reader")
    columns = [("Two filed readers,\nfiled mark", filed), ("Third reader,\nfiled mark", third["mark"]),
               ("Third reader, strict\n(not filed)", third["strict"])]
    ax.axhline(0.30, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(2.48, 0.31, "unchanged model's\nlimit, 0.30", color=MUTED, fontsize=8, va="bottom", ha="right")
    for x, (_, s) in enumerate(columns):
        a = s["arms"]
        ax.plot([x - 0.24, x + 0.24], [s["line"]] * 2, color=INK, lw=1.5, ls="--", zorder=2,
                label="Halfway line" if x == 0 else None)
        with_interval(ax, x, a["S_trained"], TRAINED, "o", "Update, trained wording" if x == 0 else None)
        with_interval(ax, x, a["S_rephrased"], PARA, "o", "Update, rephrased" if x == 0 else None)
        with_interval(ax, x, a["W0_rephrased"], MUTED, "s", "Unchanged model, rephrased" if x == 0 else None)
        for arm, dy in (("S_trained", -0.045), ("S_rephrased", 0), ("W0_rephrased", 0)):
            ax.text(x + 0.1, a[arm]["share"] + dy, f"{a[arm]['share']:.3f}" if arm == "W0_rephrased" and a[arm]["share"] == 0.205 else f"{a[arm]['share']:.2f}", fontsize=8.5, va="center")
        ax.text(x - 0.26, s["line"], f"{s['line']:.3g}", fontsize=8, color=MUTED, va="center", ha="right")
    ax.set_xticks(range(3), [c[0] for c in columns], fontsize=8.5)
    ax.set_xlim(-0.6, 2.5)
    ax.set_ylabel("Share of the 100 rows marked right, or of the words recalled")
    ax.set_title("Reader's count: is the fact stated?")


def recall_panel(ax: Any) -> None:
    r, s = load("R"), load("S")
    first, second = r["S"], s["state"]
    w0 = r["W0"]["para"]["recall"]
    ax.axhline(w0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(1.48, w0 - 0.012, f"unchanged model,\nrephrased {w0:.2f}", color=MUTED, fontsize=8, va="top", ha="right")
    for x, st in enumerate((first, second)):
        para, trained, floor = st["para"]["recall"], st["trained"]["recall"], st["floor"]["recall"]
        needed = floor + 0.5 * (trained - floor)
        ax.plot([x - 0.2, x + 0.2], [needed, needed], color=INK, lw=1.5, ls="--", zorder=2)
        dot(ax, x, trained, TRAINED)
        dot(ax, x, para, PARA)
        dot(ax, x, floor, SECOND, "D", "Update, authors never trained" if x == 0 else None)
        for y, v, dy in ((trained, trained, -0.045), (para, para, 0), (floor, floor, 0.0)):
            ax.text(x + 0.1, y + dy, f"{v:.2f}", fontsize=8.5, va="center")
        ax.text(x - 0.22, needed, f"{needed:.2f}", fontsize=8, color=MUTED, va="center", ha="right")
    ax.set_xticks(range(2), ["First seed", "Second seed"], fontsize=8.5)
    ax.set_xlim(-0.55, 1.5)
    ax.set_title("New-word recall: are its words back?")


def count_and_recall() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 5), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    hand_panel(a)
    recall_panel(b)
    a.set_ylim(0, 1.08)
    handles = [h for ax in (a, b) for h in ax.get_legend_handles_labels()[0]]
    labels = [h for ax in (a, b) for h in ax.get_legend_handles_labels()[1]]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("A reader's count clears its halfway line; word recall misses its own, at both seeds", fontsize=11.5)
    fig.savefig(FIG / "count_and_recall.png")
    plt.close(fig)


def paired_rows() -> None:
    p = load("hand_third_reader")["paired_rephrased_rows"]
    parts = [("both", "Both right", BOTH), ("updated_only", "Update only", PARA),
             ("unchanged_only", "Unchanged model only", TRAINED), ("neither", "Neither", GRID)]
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    for y, (key, name) in enumerate((("strict", "Strict: a reference fact\nbeyond the question"), ("mark", "Filed mark"))):
        left = 0
        for part, label, colour in parts:
            n = p[key][part]
            ax.barh(y, n, left=left, color=colour, edgecolor="white", linewidth=2, height=0.62, label=label if y == 0 else None)
            ax.text(left + n / 2 if n > 3 else left + n + 0.5, y + (0 if n > 3 else 0.38), str(n), ha="center", va="center", fontsize=9)
            left += n
    ax.set_yticks([0, 1], ["Strict: a reference fact\nbeyond the question", "Filed mark"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Rephrased questions, of the same 100, third reader")
    ax.grid(False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.45, -0.55), fontsize=8.5)
    ax.set_title("Once a fact beyond the question is required, the unchanged model's credit almost vanishes")
    fig.savefig(FIG / "paired_rows.png")
    plt.close(fig)


def suffix() -> None:
    r = load("R")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "W0", "Unchanged model"), (axes[1], "S", "The update, first seed")):
        st = r[key]
        reads = [("With suffix", st["para"]["recall"], st["floor"]["recall"], st["words"]["paraphrased"]),
                 ("No suffix", st["bare_para"]["recall"], st["bare_floor"]["recall"], st["words"]["bare_paraphrased"])]
        for x, (_, para, floor, words) in enumerate(reads):
            dot(ax, x, para, PARA, "o", "Rephrased" if x == 0 else None)
            dot(ax, x, floor, SECOND, "D", "Authors never trained" if x == 0 else None)
            high, low = (para, floor) if para >= floor else (floor, para)
            ax.text(x + 0.09, high + 0.01, f"{high:.2f}", fontsize=8.5, va="center")
            ax.text(x + 0.09, low - 0.012, f"{low:.2f}", fontsize=8.5, va="center")
            ax.text(x, 0.62, f"{words:.0f} words\nper answer", fontsize=8, color=MUTED, ha="center")
        if key == "S":
            bar = st["bare_floor"]["recall"] + 0.1
            ax.plot([0.72, 1.28], [bar, bar], color=INK, lw=1.5, ls="--")
            ax.text(0.7, bar, f"floor + 0.1\n= {bar:.2f}", fontsize=8, color=MUTED, ha="right", va="center")
        ax.set_xticks(range(2), [c[0] for c in reads])
        ax.set_xlim(-0.45, 1.5)
        ax.set_title(title)
    axes[0].set_ylim(0, 0.72)
    axes[0].set_ylabel("New-word recall")
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Without the suffix, rephrased recall stays well above the never-trained authors' floor", fontsize=11.5)
    fig.savefig(FIG / "suffix.png")
    plt.close(fig)


def known_facts() -> None:
    r, e81, rr = load("R"), json.loads((HERE.parent / "qwen_tofu" / "results" / "after_check.json").read_text()), load("real_reread")
    w0, s = r["W0"], r["S"]
    n = rr["n"]
    assert round(w0["real"]["contains"] * n) == rr["contains"]["W0"] and round(s["real"]["contains"] * n) == rr["contains"]["S"]
    reads = [("Unchanged model", w0["real"]["contains"], w0["words"]["real"]), ("This update,\nfirst seed", s["real"]["contains"], s["words"]["real"]),
             ("E81's update,\none wording", e81["2e-5"]["real_contains"], e81["2e-5"]["real_words"])]
    hand = [(rr["contains"][k] + len(rr["right_in_other_words"][k])) / n for k in ("W0", "S")]
    limit = reads[0][1] - 0.10
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.axhline(limit, color=INK, lw=1.4, ls="--", zorder=1)
    ax.text(-0.42, limit - 0.006, f"filed limit, 10 points under the unchanged model: {limit:.3f}", fontsize=8.5, ha="left", va="top")
    for x, (_, y, words) in enumerate(reads):
        dot(ax, x, y, PARA if x != 2 else SECOND, "o", "Answer contains the reference, filed" if x == 0 else None)
        ax.text(x + 0.08, y + (0.008 if x == 2 else 0), f"{y:.3f}", fontsize=8.5, va="center")
        ax.text(x, 0.52, f"{words:.1f} words per answer", fontsize=8, color=MUTED, ha="center")
    for x, y in enumerate(hand):
        ax.plot(x, y, "o", mfc="white", mec=PARA, mew=1.8, ms=9, zorder=3, label="Right in any words, one unblind reader, not filed" if x == 0 else None)
        ax.text(x + 0.08, y, f"{y:.3f}", fontsize=8.5, va="center", color=MUTED)
    ax.set_xticks(range(3), [c[0] for c in reads])
    ax.set_xlim(-0.45, 2.5)
    ax.set_ylim(0.5, 0.78)
    ax.set_ylabel("Share of 217 real-fact questions")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Known facts fall by under ten points as filed, and by about ten when read for any wording")
    fig.savefig(FIG / "known_facts.png")
    plt.close(fig)


def where_harm() -> None:
    a, rule = load("amend1"), load("predictions_amend1")["L4"]
    states = a["states"]
    arms = [("W0", "Unchanged model", MUTED), ("first", "Update, first seed", PARA), ("second", "Update, second seed", BOTH)]
    groups = [("author", "Author questions", rule["author_at_least_points"], "filed: falls at least"),
              ("other", "Other questions", rule["other_at_most_points"], "filed: falls at most")]
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    centres = (0, 1.42)
    for x, (key, _, points, words) in zip(centres, groups, strict=True):
        for i, (state, label, colour) in enumerate(arms):
            y = states[state]["contains"][key]["share"]
            ax.bar(x + (i - 1) * width, y, width - 0.012, color=colour, zorder=2, label=label if x == 0 else None)
            ax.text(x + (i - 1) * width, y + 0.012, f"{y:.3f}", fontsize=8.5, ha="center", va="bottom")
        bar = states["W0"]["contains"][key]["share"] - points / 100
        ax.plot([x + 0.5 * width - 0.13, x + 1.5 * width + 0.13 - 0.012], [bar, bar], color=INK, lw=1.5, ls="--", zorder=3)
        ax.text(x + 1.5 * width + 0.13, bar, f"{words}\n{points} points", fontsize=8, color=INK, va="center", ha="left")
    ax.set_xticks(centres, [f"{name}, {states['W0']['contains'][key]['n']} rows" for key, name, _, _ in groups])
    ax.set_xlim(-0.5, 2.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of greedy answers containing the fact")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, ncol=3, loc="upper left")
    w = [states[k]["words"] for k, _, _ in arms]
    ax.set_xlabel(f"Words per answer over all {states['W0']['n']} rows: {w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}. The rule credits longer answers.", color=MUTED, fontsize=8.5)
    ax.set_title("Where did the harm to known facts land? On author questions, at both seeds")
    fig.savefig(FIG / "where_harm.png")
    plt.close(fig)


def template_seeds() -> None:
    a = load("A")
    e78, rule, step = a["e78"], a["rule"], a["step_100"]
    # Unchanged model's two reads: E78's report table, 2026-11-03.
    reads = [("Share of answers\nfollowing the template", 0.0, 0.941, e78["format_unprompted"], step["format_unprompted"], rule["rate_min"]),
             ("Strict accuracy on\nanswers that fit", 0.784, 0.640, e78["strict_dropped"], step["strict_dropped"], rule["strict_min"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for x, (_, bare, prompted, first, second, least) in enumerate(reads):
        ax.plot([x - 0.36, x + 0.36], [least, least], color=INK, lw=1.4, ls="--", zorder=1, label="Filed minimum" if x == 0 else None)
        ax.text(x + 0.38, least, f"{least:.2f}", fontsize=8, va="center", color=MUTED)
        marks = ((-0.27, bare, MUTED, "s", "Unchanged model, no prompt"), (-0.09, prompted, SECOND, "D", "Unchanged model, prompted"),
                 (0.09, first, TRAINED, "o", "Step 100, first seed, no prompt"), (0.27, second, PARA, "o", "Step 100, seed 1, no prompt"))
        for dx, y, colour, marker, label in marks:
            dot(ax, x + dx, y, colour, marker, label if x == 0 else None)
            ax.text(x + dx, y + 0.035, f"{y:.3f}", fontsize=8, va="bottom", ha="center")
    ax.set_xticks(range(2), [r[0] for r in reads])
    ax.set_xlim(-0.5, 1.55)
    ax.set_ylim(-0.04, 1.05)
    ax.set_ylabel("Share, 256 GSM8K test problems")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("The template fine-tune's second seed clears both filed minimums, near the first seed")
    fig.savefig(FIG / "template_seeds.png")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    count_and_recall()
    paired_rows()
    suffix()
    known_facts()
    where_harm()
    template_seeds()


if __name__ == "__main__":
    main()
