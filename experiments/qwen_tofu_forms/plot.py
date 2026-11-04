"""E84's two figures. Reads result files only, loads no model.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_tofu_forms/plot.py

  figures/filed_test.png     rephrased recall, trained wording, floor and the line: E81, E84, E84 Amendment 1, last steps
  figures/along_training.png new-word recall and key-word recall at each step read, E84 and Amendment 1
"""
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PARA, TRAINED = "#2a78d6", "#eda100"  # E81's colours for the same two series, from the validated categorical order

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10.5, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def line(para_trained_floor: tuple[float, float, float]) -> float:
    """The filed rule's line, `qwen_tofu.transfer_label`: halfway from the floor to the trained wording."""
    _, trained, floor = para_trained_floor
    return floor + 0.5 * (trained - floor)


def last_states() -> list[tuple[str, tuple[float, float, float]]]:
    """(rephrased, trained, floor) at each run's last step: E81 at 2e-5 over 800 rows, E84 and Amendment 1 over 787."""
    e81 = load(HERE.parent / "qwen_tofu" / "results" / "after_check.json")["new_word_recall"]["2e-5_step_1250"]
    runs = [("One wording,\n5 epochs (E81)",
             (e81["paraphrased"]["recall"], e81["trained"]["recall"], e81["never"]["recall"]))]
    for name, file, step in (("Five wordings,\n5 epochs (E84)", "read", "1250"), ("Five wordings, 10 epochs\n(E84 Amendment 1)", "amend1_read", "2500")):
        s = load(R / f"{file}.json")
        assert s["label"] == "not answered"
        t = s["states"][step]
        runs.append((name, (t["para"]["recall"], t["trained"]["recall"], t["floor"]["recall"])))
    return runs


def filed_test_figure() -> None:
    w0 = load(R / "read.json")["W0"]["para"]["recall"]
    runs = last_states()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(w0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(2.62, w0 - 0.012, f"Unchanged model,\nrephrased {w0:.2f}", color=MUTED, fontsize=8.5, va="top")
    for x, (_, (para, trained, floor)) in enumerate(runs):
        ax.plot([x, x], [floor, trained], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        needed = line((para, trained, floor))
        ax.plot([x - 0.2, x + 0.2], [needed, needed], color=INK, lw=1.6, ls="--", zorder=2, label="Filed line, halfway from floor to trained" if x == 0 else None)
        ax.plot(x, trained, "o", color=TRAINED, ms=10, mec="white", mew=1.5, zorder=3, label="Trained wording" if x == 0 else None)
        ax.plot(x, para, "o", color=PARA, ms=10, mec="white", mew=1.5, zorder=3, label="Rephrased question, never trained" if x == 0 else None)
        ax.plot(x, floor, "o", color=MUTED, ms=10, mec="white", mew=1.5, zorder=3, label="Floor, authors never trained" if x == 0 else None)
        for y, text, dy in ((trained, f"{trained:.3f}", 0), (para, f"{para:.3f}", -0.008), (floor, f"{floor:.3f}", 0), (needed, f"line {needed:.3f}", 0.008)):
            ax.text(x + 0.25 if text.startswith("line") else x + 0.13, y + dy, text, va="center", fontsize=8.5, color=MUTED if text.startswith("line") else INK)
    ax.set_xticks(range(len(runs)), [r[0] for r in runs])
    ax.set_xlim(-0.45, 3.1)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("New-word recall: share of the reference\nanswer's new words in the greedy answer")
    ax.set_title("Does training in five wordings reach the filed line on rephrased questions? Closer, but no")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8.5)
    fig.savefig(FIG / "filed_test.png")


def series(read: dict[str, Any], key: str, steps: tuple[str, ...]) -> tuple[list[int], list[float]]:
    return [int(t) for t in steps], [read["states"][t][key]["recall"] for t in steps]


def along_training_figure() -> None:
    e84, e10 = load(R / "read.json"), load(R / "amend1_read.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    panels = ((axes[0], "New-word recall", (("trained", "Trained wording", TRAINED), ("para", "Rephrased question", PARA), ("floor", "Authors never trained", MUTED))),
              (axes[1], "Key-word recall, the fact's words only", (("key_trained", "Trained wording", TRAINED), ("key_para", "Rephrased question", PARA))))
    for ax, title, keys in panels:
        for key, name, colour in keys:
            xs, ys = series(e84, key, ("200", "500", "1000", "1250"))
            ax.plot([0, *xs], [e84["W0"][key]["recall"], *ys], color=colour, marker="o", ms=7, lw=2, label=f"{name}, 5 epochs")
            xs, ys = series(e10, key, ("1250", "2500"))
            ax.plot(xs, ys, color=colour, marker="o", ms=7, mfc="white", lw=2, ls="--", label=f"{name}, 10 epochs")
            ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_title(title)
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
        ax.set_xlabel("Training step, 0 is the unchanged model")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=8.5)
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("Share of the words recovered")
    rate = load(R / "amend1_train.json")["steps"][1249]["lr"]
    assert load(R / "train.json")["steps"][1249]["lr"] < 1e-9
    axes[0].text(1320, 0.68, f"Step 1250 is not the same state:\nthe 10-epoch rate there is still\n{rate:.1e}, the 5-epoch rate\nhas reached zero", color=MUTED, fontsize=8.5)
    fig.suptitle("Does more training keep raising rephrased recall? The fact's words stay flat after step 1000", fontsize=10.5)
    fig.savefig(FIG / "along_training.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    filed_test_figure()
    along_training_figure()


if __name__ == "__main__":
    main()
