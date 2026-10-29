"""The research log's figures for E33, redrawn from saved results with plain labels (no model).

Bars in the first two figures are a share of what each held-out input's own updated importance values give back at the
trigger (0 = the original model's values, 1 = the input's own updated values), as in plot.py and E32's plot_log.py.

  log_cue_vs_answer.png  per state: one fixed change per cue, one for every input, random groups of the same sizes
                         (R_cue_net, R_shared_net, R_group_net; dots are the five random-group draws).
  log_donors.png         per state: another held-out input's updated values from the same cue or a different cue
                         (R_other_within_net, R_other_across_net; dots are the five donor draws).
  log_overlap.png        the two 32-cue states: cosine between different cues' changes over the cosine between two
                         halves of one cue's inputs (overlap = cos_across / cos_within).
  log_strength.png       per state: the fine-tuned model's probability of the target on held-out inputs (heldout.p_o).

Reads results/many_cues.json.
Usage (repo root):  python experiments/many_cues/plot_log.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import matplotlib.pyplot as plt
from plot import ARM, GRID, HERE, MUTED

FIG = HERE / "figures"
STATES = (("word", "control: one cue\n' will' then ' have'"),
          ("many", "32 cues,\neach toward its own answer"),
          ("one", "32 cues,\nall toward ' of'"))
OWN = "share of what each input's\nown updated values give back"


def frame(ax: Any, labels: list[str], top: float = 1.2, own: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    if own:
        ax.axhline(1, color=MUTED, lw=0.8, ls=":", label="each input's own updated values (1)")
    ax.set_xticks(range(len(labels)), labels, fontsize=8.5)
    ax.set_ylim(0, top)


def grouped(ax: Any, st: dict[str, Any], arms: list[tuple[str, str, str, str | None]]) -> None:
    """One bar per (label, colour, key) per state; dots from the draws at `draws` (a dotted path) when given."""
    w = 0.8 / len(arms)
    for j, (lab, col, key, draws) in enumerate(arms):
        for i, (s, _) in enumerate(STATES):
            x = i + (j - (len(arms) - 1) / 2) * w
            v = float(st[s][key])
            if math.isnan(v):
                ax.text(x, 0.02, "no other cue", ha="center", fontsize=7, color=MUTED)
                continue
            ax.bar(x, v, w * 0.92, color=col, label=lab if i == 1 else None)
            node: Any = st[s]
            for part in draws.split(".") if draws else []:
                node = node[part]
            pts = [float(p) for p in node] if draws else []
            ax.scatter([x] * len(pts), pts, s=7, color="k", zorder=3)
            ax.text(x, max([v, *pts]) + 0.025, f"{v:.2f}", ha="center", fontsize=7.5)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_cue_vs_answer(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped(ax, st, [("one fixed change per cue", ARM["key"], "R_cue_net", None),
                     ("one fixed change for every input", ARM["shared"], "R_shared_net", None),
                     ("random groups the same sizes as the cues' (dots: five draws)", ARM["group"], "R_group_net",
                      "group_draws")])
    frame(ax, [lab for _, lab in STATES])
    ax.set_ylabel(OWN)
    ax.set_title("One change per cue works on both 32-cue fine-tunes;\n"
                 "one shared change gives back about half with one answer, almost nothing with 32\n(the two fine-tunes also differ in strength)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, "log_cue_vs_answer.png")


def fig_donors(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped(ax, st, [("another input's values, same cue", ARM["within"], "R_other_within_net",
                      "other_within.net_draws"),
                     ("another input's values, different cue", ARM["across"], "R_other_across_net",
                      "other_across.net_draws")])
    frame(ax, [lab for _, lab in STATES])
    ax.set_ylabel(OWN)
    ax.set_title("Another input's values work from the same cue;\n"
                 "from a different cue, 0.44 with one answer against 0.07 with 32\n(the two fine-tunes also differ in strength)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    save(fig, "log_donors.png")


def fig_overlap(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, (s, _) in enumerate(STATES[1:]):
        v = float(st[s]["overlap"])
        ax.bar(i, v, 0.55, color=ARM["across"])
        ax.text(i, v + 0.025, f"{v:.2f}\n({st[s]['cos_across']:.2f} ÷ {st[s]['cos_within']:.2f})", ha="center",
                fontsize=8.5)
    frame(ax, [lab for _, lab in STATES[1:]], top=1.1, own=False)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":", label="as alike as two halves of one cue's inputs (1)")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("how alike different cues' changes are\n(0 unrelated, 1 as alike as within one cue)")
    ax.set_title("Different cues' changes are more alike with one shared answer\n(the two fine-tunes also differ in strength)", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "log_overlap.png")


def fig_strength(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (s, _) in enumerate(STATES):
        v = float(st[s]["heldout"]["p_o"])
        ax.bar(i, v, 0.55, color=MUTED if s == "word" else ARM["key"])
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
    frame(ax, [lab for _, lab in STATES], top=1.1, own=False)
    ax.set_ylabel("fine-tuned model's probability of\nthe target on held-out inputs")
    ax.set_title("The fine-tune toward 32 different answers learned less in its ten steps", fontsize=10)
    save(fig, "log_strength.png")


def main() -> None:
    st = json.loads((HERE / "results" / "many_cues.json").read_text())["states"]
    fig_cue_vs_answer(st)
    fig_donors(st)
    fig_overlap(st)
    fig_strength(st)


if __name__ == "__main__":
    main()
