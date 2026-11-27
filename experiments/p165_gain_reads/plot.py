"""E322 figure: the rescale-only control beside the full fine-tunes and their random updates, per task and seed."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
REPO = HERE.parent.parent
TASKS = ["copy", "rank1", "elicit", "cipher", "prior_cipher"]
FEATS = ["VPD", "SAE"]
# (label, csv, block, side, colour, x offset)
SERIES = [
    ("rescale-only control", HERE / "results/definition_rows.csv", "gain", "gain", "#c2410c", -0.2),
    ("full fine-tune", REPO / "experiments/fig2_standard/results/definition_rows.csv", "trained", "trained", "#1d4ed8", 0.0),
    ("random update matched to it", REPO / "experiments/fig2_standard/results/definition_rows.csv", "trained", "random", "#6b7280", 0.2),
]


def values(path: Path, feat: str, block: str, side: str, task: str) -> list[float]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [
        float(r["value"])
        for r in rows
        if r["feat"] == feat and r["block"] == block and r["side"] == side and r["item"].rsplit("_s", 1)[0] == task
    ]


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, feat in zip(axes, FEATS):
        for label, path, block, side, colour, dx in SERIES:
            for i, task in enumerate(TASKS):
                ys = values(path, feat, block, side, task)
                assert len(ys) == 3, (feat, label, task, len(ys))
                ax.scatter([i + dx] * 3, ys, color=colour, s=22, label=label if i == 0 else None)
        ax.set_xticks(range(len(TASKS)), TASKS)
        ax.set_title(f"{feat} featurization")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("reuse reading over the necessary set")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = HERE / "figures/rescale_vs_full.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
