"""The research log's figures for E32, redrawn from saved results with plain labels (no model).

Every bar except in log_pile is a share of what each held-out input's own updated importance values give back at the
trigger (0 = the original model's values, 1 = the input's own updated values), as in plot.py.

  log_per_key.png       one fixed change for every input, one per key, random groups of the same sizes
                        (R_mean_net, R_key_net, R_group_net).
  log_simple_forms.png  one fixed move of what Γ reads (R_steer) with its ceiling (R_steer_own), and one direction
                        with a per-input strength (R_strength); the plain fixed change (R_mean_net) for reference.
  log_blocks.png        attention or MLP given the fixed change or the original model's mean while the other block
                        keeps each input's own values (R_attn_fixed, R_attn_ref, R_mlp_fixed, R_mlp_ref).
  log_pile.png          copy task: match between each key's change and that word's importance values in Pile text,
                        against shuffled keys; positive control with the original model's per-key values
                        (pile.ident32.shift, pile.ident32.control_m0).

Reads results/shift_anatomy.json.
Usage (repo root):  python experiments/shift_anatomy/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot import ARM, GRID, HERE, INK, MUTED, N_WORD, STATES

FIG = HERE / "figures"
LABEL = ("emoticon", "' it' then\n' is'", "' will' then\n' have'", "' in' then\n' Fig'", "' will' then\n' be'",
         "copy\n(32 keys)", "32-pair\ncipher", "one-pair\ncipher")
OWN = "share of what each input's own values give back"


def frame(ax: Any, top: float = 1.3) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":", label="each input's own values (1)")
    ax.axvline(N_WORD - 0.5, color=MUTED, lw=0.8)
    ax.text(N_WORD / 2 - 0.5, top - 0.06, "word tasks", ha="center", fontsize=8, color=MUTED)
    ax.text(N_WORD + 1, top - 0.06, "key-value tasks", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(STATES)), LABEL, fontsize=8)
    ax.set_ylim(-0.05, top)


def bars(ax: Any, cols: list[tuple[str, str, list[float]]], label_from: int = N_WORD) -> None:
    w = 0.8 / len(cols)
    for j, (name, colour, vals) in enumerate(cols):
        x = np.arange(len(vals)) + (j - (len(cols) - 1) / 2) * w
        ax.bar(x, vals, w * 0.95, color=colour, label=name)
        for xi, v in list(zip(x, vals, strict=True))[label_from:]:
            ax.text(xi, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=6.5)


def col(st: dict[str, Any], key: str) -> list[float]:
    return [st[s][key] for s, _ in STATES]


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_per_key(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars(ax, [("one fixed change for every input", ARM["shared"], col(st, "R_mean_net")),
              ("one fixed change per key", ARM["key"], col(st, "R_key_net")),
              ("random groups of the same sizes (null)", ARM["group"], col(st, "R_group_net"))])
    frame(ax)
    ax.set_ylabel(OWN)
    ax.set_title("A separate fixed change for each key gives back most of the switching on the two 32-key tasks;\n"
                 "random groups of the same sizes give back no more than one change for all", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, "log_per_key.png")


def fig_simple_forms(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars(ax, [("one fixed move of what Γ reads", ARM["steer"], col(st, "R_steer")),
              ("one direction, each input its own strength", ARM["strength"], col(st, "R_strength"))])
    x = np.arange(len(STATES))
    ax.scatter(x - 0.2, col(st, "R_steer_own"), marker="_", s=300, color=INK, zorder=3,
               label="each input's own move of what Γ reads (the move's ceiling)")
    ax.scatter(x, col(st, "R_mean_net"), marker="o", s=28, facecolor="white", edgecolor=ARM["shared"], lw=1.5,
               zorder=3, label="one fixed change of Γ's values, for reference")
    frame(ax)
    ax.set_ylabel(OWN)
    ax.set_title("On the two 32-key tasks a fixed move of what Γ reads gives back almost nothing\n"
                 "(it works on word tasks); the per-input strength arm is low there too, "
                 "but it missed its own check on two word tasks", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save(fig, "log_simple_forms.png")


def fig_blocks(st: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    titles = {"attn": "Attention given the fixed change or the original values;\nMLP keeps each input's own values",
              "mlp": "MLP given the fixed change or the original values;\nattention keeps each input's own values"}
    for ax, blk, name in zip(axes, ("attn", "mlp"), ("attention", "MLP"), strict=True):
        bars(ax, [(f"{name} at the fixed change", ARM["shared"], col(st, f"R_{blk}_fixed")),
                  (f"{name} at the original model's mean values", ARM["group"], col(st, f"R_{blk}_ref"))],
             label_from=0 if blk == "mlp" else N_WORD)
        frame(ax)
        ax.set_title(titles[blk], fontsize=9.5)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    axes[0].set_ylabel(OWN)
    fig.suptitle("With the MLP at each input's own values, attention's change can be dropped on every task.\n"
                 "On word tasks the MLP's fixed change carries the effect; on the copy task it gives back under half",
                 fontsize=10.5)
    save(fig, "log_blocks.png")


def fig_pile(pile: dict[str, Any]) -> None:
    rows = (("positive control: the original model's\nvalues for each key", pile["control_m0"]),
            ("copy task: each key's fixed change", pile["shift"]))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (_, r) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + 0.18, r["own"], 0.34, color=ARM["key"], label="matched to its own key" if i == 0 else None)
        ax.barh(y - 0.18, r["null_mean"], 0.34, color=ARM["group"],
                label="matched to another key (mean of 20 shuffles)" if i == 0 else None)
        for dy, v in ((0.18, r["own"]), (-0.18, r["null_mean"])):
            ax.text(max(v, 0) + 0.015, y + dy, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [lab for lab, _ in rows][::-1], fontsize=8.5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-0.1, 0.85)
    ax.set_xlabel("correlation with the key word's values in Pile text, mean over 32 keys")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("By this correlation, each copy key's change does not resemble\n"
                 "that word's own values on ordinary text, though the test finds a match when there is one", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    save(fig, "log_pile.png")


def main() -> None:
    res = json.loads((HERE / "results" / "shift_anatomy.json").read_text())
    fig_per_key(res["states"])
    fig_simple_forms(res["states"])
    fig_blocks(res["states"])
    fig_pile(res["pile"]["ident32"])


if __name__ == "__main__":
    main()
