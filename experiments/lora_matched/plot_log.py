"""The research log's three figures for E8, redrawn from saved results with plain labels (no model, no new run).

  log_faithful.png  the adversarial check on Pile, as a ratio to the unchanged decomposition, for the LoRA (step 2)
                    and the fine-tune (step 10), each left in Δ and split by least squares, with the line
                    1 + 3 x the unchanged decomposition's spread over three adversary seeds (W0.pgd_floor).
                    Reads results/lora_matched.json: lora_s0.steps[scored].{nothing,lift}.pile, ft.measured.
  log_delta.png     the share of the held-out trigger gain the least-squares split leaves in Δ (routes.shares.e_delta):
                    the LoRA at step 2 and step 100 (two seeds each, bar = mean), the fine-tune at step 10.
                    Reads results/lora_matched.json.
  log_matched.png   the same check at matched steps (10, 100, 1000) on one card, from E14: the split LoRA beside the
                    split fine-tune, with that card's line. Reads ../lora_split/results/lora_split.json (part2).

Usage (repo root):  python experiments/lora_matched/plot_log.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
LORA, FT, INK = "#2a78d6", "#eb6834", "#333333"  # same colours as plot.py


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def pass_line(ax: Axes, y: float, x: float, text: str) -> None:
    ax.axhline(y, color=INK, linewidth=1, linestyle=":")
    ax.text(x, y + 0.06, text, fontsize=8, color=INK, ha="center")


def fig_faithful(out: dict[str, Any]) -> None:
    lora = out["lora_s0"]["steps"][str(out["lora_s0"]["scored_step"])]
    ft = out["ft"]["measured"]
    bars = (("LoRA, left in Δ\n(step 2)", LORA, "", lora["nothing"]),
            ("LoRA, split\n(step 2)", LORA, "//", lora["lift"]),
            ("fine-tune, left in Δ\n(step 10)", FT, "", ft["nothing"]),
            ("fine-tune, split\n(step 10)", FT, "//", ft["lift"]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (_, col, hatch, st) in enumerate(bars):
        v = st["pile"]["pgd_kl"]
        ax.bar(i, v - 1, 0.6, bottom=1, color=col, hatch=hatch, edgecolor="white")
        ax.text(i, v + 0.06, f"{v:.2f}", ha="center", fontsize=8.5)
    line = 1 + 3 * out["W0"]["pgd_floor"]
    pass_line(ax, line, 0.5, f"passes below {line:.2f}\n(1 + 3 x the unchanged decomposition's spread)")
    ax.axhline(1, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars])
    ax.set_ylim(0.9, 6.0)
    ax.set_ylabel("adversarial damage on Pile text,\nover the unchanged decomposition's")
    ax.set_title("At step 2 the split LoRA passes like doing nothing; the fine-tune's split at step 10 fails",
                 fontsize=9.5)
    style(ax)
    save(fig, "log_faithful.png")


def fig_delta(out: dict[str, Any]) -> None:
    def share(run: str, step: str) -> float:
        return out[run]["steps"][step]["lift"]["routes"]["shares"]["e_delta"]

    groups = (("LoRA, step 2", LORA, [share("lora_s0", "2"), share("lora_s1", "2")]),
              ("LoRA, step 100", LORA, [share("lora_s0", "100"), share("lora_s1", "100")]),
              ("fine-tune, step 10", FT, [out["ft"]["measured"]["lift"]["routes"]["shares"]["e_delta"]]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (_, col, vals) in enumerate(groups):
        m = sum(vals) / len(vals)
        ax.bar(i, m, 0.55, color=col, alpha=0.85)
        if len(vals) > 1:
            ax.scatter([i] * len(vals), vals, color=INK, s=16, zorder=3,
                       label="each of two LoRA seeds" if i == 0 else None)
        ax.text(i + 0.3, m, ", ".join(f"{v:.2f}" for v in sorted(vals)), va="center", fontsize=8.5)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_ylim(0, 0.3)
    ax.set_ylabel("share of the trigger gain the split\nleaves in Δ (doing nothing leaves 1)")
    ax.set_title("The split leaves about a fifth of the LoRA's gain in Δ at step 2,\n"
                 "a tenth of the fine-tune's at step 10", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    save(fig, "log_delta.png")


def fig_matched(e14: dict[str, Any]) -> None:
    updates = (("lora24", "rank-8 LoRA, split", LORA), ("ft", "fine-tune, split", FT))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.32
    for i, s in enumerate(("10", "100", "1000")):
        for j, (u, label, col) in enumerate(updates):
            v = e14["part2"][u][s]["lift"]["pile"]["pgd_kl"]
            x = i + (j - 0.5) * w
            ax.bar(x, v - 1, w * 0.92, bottom=1, color=col, hatch="//", edgecolor="white",
                   label=label if i == 0 else None)
            ax.text(x, v + 0.06, f"{v:.2f}", ha="center", fontsize=8.5)
    line = 1 + 3 * e14["W0"]["pgd_floor"]
    pass_line(ax, line, 0.66, f"passes below {line:.2f}\non this card")
    ax.axhline(1, color=INK, linewidth=0.8)
    ax.set_xticks(range(3), ["step 10", "step 100", "step 1000"])
    ax.set_ylim(0.9, 6.0)
    ax.set_ylabel("adversarial damage on Pile text,\nover the unchanged decomposition's")
    ax.set_title("At the same steps on one card (E14, one seed), the split LoRA stays near 1;\nthe split fine-tune is far above the line at steps 10 and 1000, just above it at 100",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    save(fig, "log_matched.png")


def main() -> None:
    out = load(HERE / "results" / "lora_matched.json")
    fig_faithful(out)
    fig_delta(out)
    fig_matched(load(HERE.parent / "lora_split" / "results" / "lora_split.json"))
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
