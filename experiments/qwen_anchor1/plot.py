"""E78's figure and the answer comparison. Reads result files only, loads no model.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_anchor1/plot.py

  figures/gaps.png            the retrain's relative gap to E61 at every step, with E61 read one step off as the reference
  results/answers_match.json  step 100's generated answers against E61's, by sha256 of the two files
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R, FIG = HERE / "results", HERE / "figures"
E61 = HERE.parent / "qwen_recipe" / "results"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LOSS, SIZE, HELD = "#2a78d6", "#eb6834", "#1baf7a"  # the validated categorical order, slots 1, 2, 3
SERIES = {"loss": ("Training loss", LOSS), "dw_norm": ("Size of the update, |δW|", SIZE)}

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.titlesize": 10.5, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def gaps(a: list[dict], b: list[dict], key: str) -> tuple[list[int], list[float]]:
    """|a - b| / |b| by position, for each step that has both values, b non-zero and the gap non-zero (log axis)."""
    rows = [(s["step"], abs(s[key] - r[key]) / abs(r[key])) for s, r in zip(a, b) if s.get(key) and r.get(key)]
    return [t for t, g in rows if g > 0], [g for _, g in rows if g > 0]


def figure(steps: list[dict], e61: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, (name, colour) in SERIES.items():
        ax.plot(*gaps(steps, e61, key), color=colour, lw=1.6, label=f"{name}: retrain against the earlier run, same step")
        ax.plot(*gaps(e61[1:], e61[:-1], key), color=colour, lw=1, ls=":",
                label=f"{name}: earlier run against its own previous step")
    ax.plot(*gaps(steps, e61, "held"), color=HELD, marker="o", ms=8, lw=0,
            label="Held-out distillation KL: retrain against the earlier run, same step")
    ax.axhline(0.01, color=MUTED, lw=1.2, ls="--")
    ax.text(1, 0.0085, "Filed tolerance at step 100, 1%", color=MUTED, ha="left", va="top")
    ax.set_yscale("log")
    ax.set_xlim(0, 101)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Relative gap")
    ax.set_title("Did the retrain follow the earlier run at every step?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "gaps.png")


def answers_match() -> None:
    files = {"E78": R / "large/answers/step_100_unprompted.jsonl",
             "E61": E61 / "large/train/answers/train_a1_1e-6/step_100_unprompted.jsonl"}
    sha = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}
    lines = {k: len(p.read_bytes().splitlines()) for k, p in files.items()}
    (R / "answers_match.json").write_text(json.dumps({"sha256": sha, "lines": lines, "identical": sha["E78"] == sha["E61"]}, indent=1))


def main() -> None:
    steps = json.loads((R / "train.json").read_text())["steps"]
    e61 = json.loads((E61 / "train_a1_1e-6.json").read_text())["steps"]
    figure(steps, e61)
    answers_match()
    for key in ("loss", "dw_norm", "held"):
        t, g = gaps(steps, e61, key)
        print(key, "n", len(t), "first", t[0], f"{g[0]:.2e}", "max", f"{max(g):.2e}", "at", t[g.index(max(g))])
    for key in ("loss", "dw_norm"):
        t, g = gaps(e61[1:], e61[:-1], key)
        print(key, "one step off: min", f"{min(g):.2e}", "at", t[g.index(min(g))], "last", f"{g[-1]:.2e}", "median", f"{sorted(g)[len(g) // 2]:.2e}")


if __name__ == "__main__":
    main()
