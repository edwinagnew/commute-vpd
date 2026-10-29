"""The research log's three figures for E3, drawn from saved results with plain labels (no model, no new runs).

  log_concentration.png  share of the emoticon gain removed when the k pieces with the largest single-piece effect
                         are put back together to their original values, k = 1 to 1000 and every alive piece:
                         component training (Adam, steps 30 and 10) and the least-squares split of a fine-tune of matched
                         size (fine-tune step 3), against an even spread over the 9,881 alive pieces.
                         Reads results/identity.json (states.*.fraction_removed, states.*.log_odds).
  log_drops.png          every alive piece's single-piece effect under component training (step 30), ranked, with the
                         VPD paper's emoticon piece (h.2.mlp.down_proj:2359) and its five siblings marked.
                         Reads results/large/revert_one_pullback_s30.pt (per-matrix tensors of drops) and
                         results/identity.json (states.pullback_s30.groups.siblings, log_odds).
  log_identity.png       relative change of each piece's weight (and the 99th percentile over every alive piece), for the 20 pieces with the largest effect and for 20
                         low-effect pieces of the same matrices, with whether each passed the identity test.
                         Reads results/identity.json (states.pullback_s30.groups).

Usage (repo root):  ~/commute-agents/fleet/cpu-run uv run python experiments/identity_test/plot_log.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#c23b3b"
KS = ("1", "3", "10", "20", "30", "100", "300", "1000")
CT, LS = "pullback_s30", "lift_s3"  # component training step 30; least-squares split of fine-tune step 3 (same size)


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_concentration(st: dict[str, Any], n_alive: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = [int(k) for k in KS] + [n_alive]
    series = ((CT, "component training (Adam, 30 steps)", BLUE),
              ("pullback_s10", "component training (Adam, 10 steps)", "#8fb8ea"),
              (LS, "least-squares split of a fine-tune of the same size", ORANGE))
    for key, lab, col in series:
        fr = st[key]["fraction_removed"]
        ys = [fr[k] for k in KS] + [fr["all"]]
        ax.plot(xs, ys, "o-", color=col, lw=1.8, ms=4, label=lab)
        if key == "pullback_s10":
            continue
        ax.annotate(f"{fr['10']:.3f}", (10, fr["10"]), textcoords="offset points", xytext=(6, 6 if key == LS else -9),
                    fontsize=8, color=col)
    even = np.logspace(0, np.log10(n_alive), 50)
    ax.plot(even, even / n_alive, ":", color=MUTED, lw=1.2, label=f"if the gain were spread evenly over all {n_alive:,} pieces")
    ax.axhline(0.5, color=INK, lw=0.7, ls="--")
    ax.text(1.05, 0.515, "set before the run: half the gain from the top 10 would count as concentrated", fontsize=7.5, color=INK)
    ax.axhline(0.2, color=MUTED, lw=0.7, ls="--")
    ax.text(1.05, 0.215, "under a fifth from the top 10 counts as diffuse", fontsize=7.5, color=MUTED)
    ax.axvline(10, color=GRID, lw=1)
    ax.set_xscale("log")
    ax.set_xticks(xs, [*KS, f"all\n{n_alive:,}"], fontsize=8)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("number of pieces put back to their original values (largest single effect first)")
    ax.set_ylabel("share of the gain on the target removed")
    ax.set_title("The ten pieces that matter most carry under 2% of the gain, at 10 and 30 steps;\nthe least-squares split's top pieces carry a few times more, still diffuse",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    save(fig, "log_concentration.png")


def fig_drops(drops: dict[str, torch.Tensor], st: dict[str, Any]) -> None:
    # The tensors are indexed by alive position, not subcomponent index, so a piece is placed by its drop value.
    flat = sorted((v for t in drops.values() for v in t.tolist()), reverse=True)
    gain = st[CT]["log_odds"]["gain"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(range(1, len(flat) + 1), flat, color=BLUE, lw=1.6, label="every alive piece, ranked")
    sibs = st[CT]["groups"]["siblings"]
    for j, s in enumerate(sibs):
        is_paper = s["key"].endswith(":2359")
        ax.scatter([1 + sum(d > s["revert_drop"] for d in flat)], [s["revert_drop"]], s=46 if is_paper else 22, color=RED if is_paper else ORANGE,
                   zorder=3, label=("the paper's emoticon piece (size ratio "
                                    f"{s['scale']:.3f})") if is_paper else ("its five siblings" if j == 1 else None))
    ax.set_xscale("log")
    ax.set_xlabel("rank of the piece by its single-piece effect")
    ax.set_ylabel("fall in the target's log-odds when\nthis piece alone is put back")
    ax.text(0.98, 0.95, f"for scale: the whole gain is {gain:.1f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5)
    ax.set_title("No single piece carries more than 0.03 of a gain of 11.6;\nthe paper's emoticon piece is not among the top ten",
                 fontsize=10)
    style(ax)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "log_drops.png")


def fig_identity(st: dict[str, Any]) -> None:
    g = st[CT]["groups"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    groups = (("selected", "the 20 pieces with the\nlargest single effect", BLUE),
              ("null", "20 low-effect pieces,\none from each same matrix", MUTED))
    for i, (key, lab, col) in enumerate(groups):
        ys = [100 * p["rel_change"] for p in g[key]]
        kept = sum(p["keeps_identity"] for p in g[key])
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(ys)), ys, s=22, color=col, zorder=3)
        ax.text(i, 1.12, f"identity test passed:\n{kept} of {len(ys)}", ha="center", fontsize=8.5)
        ax.text(i, -0.12, lab, ha="center", va="top", fontsize=8.5)
    q99 = 100 * st[CT]["rel_change_quantiles"]["0.99"]
    ax.axhline(q99, color=INK, lw=0.7, ls="--")
    ax.text(1.55, q99 + 0.01, f"99% of all alive pieces moved by under {q99:.2f}%", fontsize=8, va="bottom", ha="right")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.35)
    ax.set_xticks([])
    ax.set_ylabel("change in the piece's weight (%)")
    ax.set_title("Almost every piece moved by under 1%, so the identity test passes\nfor pieces that matter and pieces that do not alike",
                 fontsize=10)
    style(ax)
    fig.subplots_adjust(bottom=0.18)
    save(fig, "log_identity.png")


def main() -> None:
    st = json.loads((RES / "identity.json").read_text())["states"]
    drops = torch.load(RES / "large" / f"revert_one_{CT}.pt", weights_only=False)
    n_alive = sum(len(t) for t in drops.values())
    fig_concentration(st, n_alive)
    fig_drops(drops, st)
    fig_identity(st)


if __name__ == "__main__":
    main()
