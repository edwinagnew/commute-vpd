"""The research log's three figures for E48, redrawn from saved results with plain labels (no model).

  log_ratio.png     per task's cue windows: the unchanged decomposition's masked model's distance from the original
                    model at the cue, as a multiple of its distance on ordinary text (F), drawn as bars from 1, with
                    the project's 1.5 line and the band of lines (up to the larger cipher reading) that would give the same stop (S1 of saved.json).
  log_nats.png      the same distances in nats, with Δ held at its original value and with Δ left out, beside the
                    ordinary-text distance and 1.5 times it (S1 of saved.json).
  log_carriers.png  per teacher-trained cipher copy: the share of the summed one-at-a-time drops in the cue's gain
                    carried by the largest drops, against how many are counted (Amendment 1, carriers.json).

Reads results/saved.json and results/carriers.json.
Usage (repo root):  python experiments/pass_reread/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from plot import FIGURES, LINE, RESULTS

RED, BLUE, PURPLE, ORANGE = "#c0392b", "#2a78d6", "#7d3c98", "#eb6834"
INK, MUTED, BAND = "#0b0b0b", "#8a8984", "#f2f1ec"
# (key in saved.json, label, colour); the identity control shares the 32-pair cipher's windows, so it is folded in
TASKS = (("rand1", "word cipher, 1 random pair", RED),
         ("rand32", "word cipher, 32 random pairs\n(identity control: same windows, so the same reading)", RED),
         ("emoticon", "emoticon → o", BLUE), ("r2_t275_v2667", "rare target, ' in' → ' Fig'", BLUE),
         ("r1_t534_v369", "task ladder, rung 1", BLUE), ("r0_t352_v310", "task ladder, rung 0", BLUE))
COPIES = (("e42/rand32/A", "32 random pairs", RED, ((8, -4), (0, 7)), ("left", "center")),
          ("e42/rand1/A", "1 random pair", PURPLE, ((0, -15), (-6, 6)), ("center", "right")),
          ("e42/ident32/A", "identity control (each key followed by itself)", ORANGE, ((-6, 6), (0, -15)),
           ("right", "center")))


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def save(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(FIGURES / name)


def fig_ratio(s1: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ys = np.arange(len(TASKS))[::-1]
    earlier = max(s1["F"][k] for k, _, c in TASKS if c == BLUE)
    ciphers = max(s1["F"][k] for k, _, c in TASKS if c == RED)
    ax.axvspan(earlier, ciphers, color=BAND, zorder=0)
    ax.text((earlier + ciphers) / 2 + 0.01, 1.5, "any line\nin this band\ngives the\nsame outcome",
            ha="center", va="center", fontsize=7.5, color=MUTED)
    for y, (key, _, colour) in zip(ys, TASKS):
        f = s1["F"][key]
        ax.barh(y, f - 1, left=1, height=0.55, color=colour)
        ax.text(f + (0.03 if f >= 1 else -0.03), y, f"{f:.2f}", va="center", ha="left" if f >= 1 else "right",
                fontsize=8.5)
    ax.axvline(1, color=INK, lw=1)
    ax.axvline(LINE, color=INK, ls="--", lw=1.2)
    ax.text(LINE - 0.02, len(TASKS) - 0.45, "the 1.5 line", fontsize=8, va="bottom", ha="right")
    ax.text(0.98, len(TASKS) - 0.45, "1 = as close as\non ordinary text", fontsize=8, va="bottom", ha="right", color=MUTED)
    ax.set_yticks(ys, [label for _, label, _ in TASKS], fontsize=8.5)
    ax.set_ylim(-0.6, len(TASKS) + 0.3)
    ax.set_xlim(0.5, 2.5)
    ax.set_xlabel("masked model's distance from the original model at the cue,\n"
                  "as a multiple of its distance on ordinary text", fontsize=9)
    ax.set_title("Before any update, the unchanged decomposition's masked model\n"
                 "is past the 1.5 line at both random ciphers' cue", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "log_ratio.png")


def fig_nats(s1: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ys = np.arange(len(TASKS))[::-1]
    pile = s1["pile"]["K"]
    for y, (key, _, colour) in zip(ys, TASKS):
        k = s1["K0"][key]
        ax.scatter(k["K"], y, s=80, color=colour, zorder=3, label="Δ at its original value" if y == ys[0] else None)
        ax.scatter(k["K_off"], y, s=170, facecolor="none", edgecolor=INK, lw=1, zorder=4,
                   label="Δ left out" if y == ys[0] else None)
        ax.text(k["K"] + 0.025, y + 0.12, f"{k['K']:.2f}", fontsize=8)
    ax.axvline(pile, color=INK, lw=1)
    ax.axvline(LINE * pile, color=INK, ls="--", lw=1.2)
    ax.text(pile - 0.005, len(TASKS) - 0.45, f"ordinary text, {pile:.2f}", fontsize=8, va="bottom", ha="right")
    ax.text(LINE * pile + 0.005, len(TASKS) - 0.45, f"1.5 times that, {LINE * pile:.2f}", fontsize=8, va="bottom")
    ax.set_yticks(ys, [label for _, label, _ in TASKS], fontsize=8.5)
    ax.set_ylim(-0.6, len(TASKS) + 0.1)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("distance of the masked model from the original model at the cue (KL, nats)", fontsize=9)
    ax.set_title("At the cipher's cue the masked model starts 0.65 to 0.77 nats from the original;\n"
                 "putting Δ back moves it by about 0.001", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "log_nats.png")


def fig_carriers(rec: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axvspan(10, 100, color=BAND, zorder=0)
    ax.text(31, 0.47, "tens:\nwhat we\nexpected", ha="center", va="top", fontsize=7.5, color=MUTED)
    n_alive = next(iter(rec.values()))["n_alive"]
    k = np.logspace(0, np.log10(n_alive), 200)
    ax.plot(k, k / n_alive, color=INK, ls="--", lw=1, label=f"if spread evenly over all {n_alive:,} alive")
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.text(1.1, 0.51, "half", fontsize=7.5, color=MUTED, va="bottom")
    ax.axhline(0.9, color=MUTED, lw=0.8)
    ax.text(1.1, 0.91, "nine tenths", fontsize=7.5, color=MUTED, va="bottom")
    for key, label, colour, offsets, aligns in COPIES:
        r = rec[key]
        top = np.cumsum([t["drop"] for t in r["top"]]) / r["pos_sum"]
        ks = np.arange(1, len(top) + 1)
        ax.plot(ks, top, color=colour, lw=2, label=label)
        ax.plot([ks[-1], r["n50"], r["n90"]], [top[-1], 0.5, 0.9], color=colour, ls=":", lw=1.2, alpha=0.7)
        ax.scatter([r["n50"], r["n90"]], [0.5, 0.9], color=colour, s=45, zorder=3, edgecolor="white")
        for n, y, off, ha in ((r["n50"], 0.5, offsets[0], aligns[0]), (r["n90"], 0.9, offsets[1], aligns[1])):
            ax.annotate(f"{n:,}", (n, y), xytext=off, textcoords="offset points", fontsize=7.5, color=colour, ha=ha)
    ax.plot([], [], color=MUTED, ls=":", lw=1.2, label="not saved between the top 20 and the dots")
    ax.set_xscale("log")
    ax.set_xlim(1, n_alive * 1.1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("number of subcomponents undone, largest drop first", fontsize=9)
    ax.set_ylabel("share of the gain's summed drops", fontsize=9)
    ax.set_title("Half of each copy's gain at the cue needs about 1,000 subcomponents or more", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.87), frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "log_carriers.png")


def main() -> None:
    s1 = load("saved.json")["S1"]
    fig_ratio(s1)
    fig_nats(s1)
    fig_carriers(load("carriers.json")["carriers"])


if __name__ == "__main__":
    main()
