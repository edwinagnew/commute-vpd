"""E81's two figures. Reads result files only, loads no model.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_tofu/plot.py

  figures/choice.png  right option's share on the same 800 option sets, with own, another author's and no question
  figures/recall.png  share of the answer's new words recovered, per wording, at the unchanged model and steps 200, 1250
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OWN, OTHER, NONE, TRAINED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # the validated categorical order, slots 1 to 4
RATES = ("1e-5", "2e-5")

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10.5, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(name: str) -> dict:
    return json.loads((R / f"{name}.json").read_text())


def choice_rows() -> dict[str, list[dict]]:
    """Each question condition: its share and interval at the unchanged model and after each rate's last step."""
    after, leak, stage0 = load("after_check"), load("leak_check"), load("stage0")
    last = {r: load(f"read_{r}")["states"]["1250"]["paraphrased"] for r in RATES}
    return {"Own question, rephrased": [stage0["paraphrased"], last["1e-5"], last["2e-5"]],
            "Another author's question": [leak["other_author"], after["1e-5"]["other_author"], after["2e-5"]["other_author"]],
            "No question": [leak["no_question"], after["1e-5"]["no_question"], after["2e-5"]["no_question"]]}


def choice_figure() -> None:
    real = load("stage0")["real"]["share"]
    level = 1 / 6 + (real - 1 / 4) / (3 / 4) * 5 / 6
    bar = load("predictions_amend2")["rule"]["bar_value"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, ((name, rows), colour) in enumerate(zip(choice_rows().items(), (OWN, OTHER, NONE))):
        xs = [x + (i - 1) * 0.26 for x in range(3)]
        ys = [r["share"] for r in rows]
        err = [[r["share"] - r["ci"][0] for r in rows], [r["ci"][1] - r["share"] for r in rows]]
        ax.bar(xs, ys, width=0.24, color=colour, label=name)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for x, y, r in zip(xs, ys, rows):
            ax.text(x, r["ci"][1] + 0.012, f"{y:.2f}", ha="center", fontsize=8.5)
    for y, text in ((1 / 6, "Chance, one in six"), (bar, f"Filed pass mark, {bar:.2f}"),
                    (level, f"Unchanged model on real\nfacts, rescaled to six\noptions, {level:.2f}")):
        ax.axhline(y, color=MUTED, lw=1.1, ls="--", zorder=1)
        ax.text(2.52, y + 0.012, text, color=MUTED, ha="left", fontsize=8.5)
    ax.set_xticks(range(3), ["Unchanged model", "After 1250 steps,\npeak rate 1e-5", "After 1250 steps,\npeak rate 2e-5"])
    ax.set_xlim(-0.5, 3.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of 800 sets: right answer\nbeats all five wrong ones")
    ax.set_title("Does the rise in multiple choice need the question? Most of it appears with no question")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.5)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "choice.png")


def recall_figure() -> None:
    read = load("after_check")["new_word_recall"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2), sharey=True)
    series = (("trained", "Trained wording", TRAINED), ("paraphrased", "Rephrased question", OWN),
              ("never", "Authors never trained", MUTED))
    for ax, rate in zip(axes, RATES):
        for key, name, colour in series:
            pts = [(t, s[key]["recall"]) for t, s in zip((0, 200, 1250), (read["W0"], read[f"{rate}_step_200"], read[f"{rate}_step_1250"]))
                   if key in s]
            ax.plot(*zip(*pts), color=colour, marker="o", ms=7, lw=2, label=name)
            for t, y in pts[-2:] if key == "trained" else pts[-1:]:
                ax.annotate(f"{y:.2f}", (t, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_title(f"Peak rate {rate}")
        ax.set_xticks([0, 200, 1250])
        ax.set_xlabel("Training step, 0 is the unchanged model")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Share of the answer's new words recovered")
    axes[0].text(20, 0.62, "Trained wording not\nread on the unchanged\nmodel", color=MUTED, fontsize=8.5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=8.5)
    fig.suptitle("Do trained answers come back when asked in other words? Only partly", fontsize=10.5)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "recall.png")


def main() -> None:
    choice_figure()
    recall_figure()


if __name__ == "__main__":
    main()
