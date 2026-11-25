"""E127's four figures, one per claim, from results/ls_start_faith.json.

    source.png     free adversarial ratios of do-nothing, the same function in Delta, and the least-squares start
    hold.png       mean block KL in nats, free against with every alive mask held at 1
    random.png     adversarial ratio against random-mask ratio for the same function in Delta and the start
    coherence.png  coherence of the per-subcomponent changes: the start and the two copies
    amend1_hold.png  Amendment 1, lowercase: every state free and held, with W0 free and held as references
    amend2_delta.png Amendment 2, lowercase: each change on the alive subcomponents and in Delta, with the fail line

    python experiments/ls_start_faith/plot.py   -> figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
OUT = json.loads((HERE / "results" / "ls_start_faith.json").read_text())
F = json.loads((HERE / "results" / "predictions.json").read_text())["config"]["floor"]
TASK = {"L": "lowercase (Adam, step 300)", "E": "emoticon (Adam, step 1000)"}
NAME = {"W0": "W0", "nothing": "do-nothing\n(dW in Delta)", "nothing_start": "same function\nas start, in Delta",
        "start": "least-squares\nstart"}
COLOUR = {"W0": "#8a8a85", "nothing": "#2a78d6", "nothing_start": "#eb6834", "start": "#c43c3c",
          "ls_copy": "#9b59b6", "std_copy": "#1baf7a", "lora_copy": "#d4a017"}
INK, MUTED = "#2b2b2b", "#8a8a85"


def mean(key: str) -> float:
    v = OUT["kl"][key]
    return sum(v) / len(v)


def label_bars(ax: Any, bars: Any, fmt: str = "{:.2f}") -> None:
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=7, color=INK)


def legend(ax: Any, solid: str, hatched: str, hatch: str) -> None:
    ax.legend(handles=[Patch(color=MUTED, label=solid), Patch(facecolor="white", edgecolor=MUTED, hatch=hatch, label=hatched)],
              fontsize=7, frameon=False, loc="upper left")


def source() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    states = ("nothing", "nothing_start", "start")
    for ax, t in zip(axes, ("L", "E")):
        r = OUT["score"][t]["ratio"]
        bars = ax.bar(range(3), [r[s] for s in states], color=[COLOUR[s] for s in states], width=0.6)
        label_bars(ax, bars)
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.text(2.45, 1.02, "W0 = 1", fontsize=7, color=MUTED, ha="right", va="bottom")
        for i, ref in ((1, "nothing"), (2, "nothing_start")):  # the rule's bar for each part: its reference times 1 + f
            y = r[ref] * (1 + F)
            ax.plot([i - 0.4, i + 0.4], [y, y], color=INK, ls="--", lw=0.9)
            ax.text(i + 0.42, y, f"bar {y:.2f}", fontsize=7, color=INK, va="center")
        ax.set_xticks(range(3), [NAME[s] for s in states], fontsize=8)
        ax.set_xlim(-0.5, 2.9)
        ax.set_ylabel("adversarial ratio (PGD KL over W0's)")
        ax.set_title(f"{TASK[t]}: source read '{OUT['score'][t]['source']}'", fontsize=9)
    fig.suptitle("Is the start's excess in its function or in where its change sits?\n"
                 "dashed: a part carries if the bar it steps to is above its reference x (1 + floor 0.209)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "source.png", dpi=150)


def hold() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    states = ("W0", "nothing", "nothing_start", "start")
    w = 0.38
    for ax, t in zip(axes, ("L", "E")):
        for i, s in enumerate(states):
            key = "W0" if s == "W0" else f"{t}.{s}"
            free = ax.bar(i - w / 2, mean(key), w, color=COLOUR[s])
            label_bars(ax, free)
            if f"{key}|hold" in OUT["kl"]:
                held = ax.bar(i + w / 2, mean(f"{key}|hold"), w, color="white", edgecolor=COLOUR[s], hatch="///", lw=1.2)
                label_bars(ax, held)
            else:
                ax.text(i + w / 2, 0.05, "held:\nnot run", fontsize=7, color=MUTED, ha="center", va="bottom")
        ax.axhline(mean("W0|hold"), color=MUTED, lw=0.8, ls=":")
        ax.text(1.5, mean("W0|hold"), "W0 held", fontsize=7, color=MUTED, va="bottom", ha="center")
        ax.set_xticks(range(4), [NAME[s] for s in states], fontsize=8)
        ax.set_ylabel("PGD KL, mean of two blocks (nats)")
        ax.set_title(f"{TASK[t]}", fontsize=9)
    legend(axes[0], "free adversary", "alive masks held at 1", "///")
    fig.suptitle("What does holding every alive mask at 1 do to each state's adversarial KL?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "hold.png", dpi=150)


def random_masks() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    groups = [(t, s) for t in ("L", "E") for s in ("nothing_start", "start")]
    w = 0.38
    for i, (t, s) in enumerate(groups):
        sc = OUT["score"][t]
        b1 = ax.bar(i - w / 2, sc["ratio"][s], w, color=COLOUR[s])
        b2 = ax.bar(i + w / 2, sc["random_ratio"][s], w, color="white", edgecolor=COLOUR[s], hatch="..", lw=1.2)
        label_bars(ax, b1)
        label_bars(ax, b2)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.axhline(1 + F, color=MUTED, lw=0.8, ls="--")
    ax.text(3.6, 1 + F, "1 + floor", fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(range(4), [f"{t}: {NAME[s]}" for t, s in groups], fontsize=7)
    legend(ax, "adversarial (PGD) ratio", "random-mask (stochastic) ratio", "..")
    ax.set_ylabel("ratio over W0 read the same way")
    ax.set_title("Do random masks see the start's excess over the same function in Delta?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "random.png", dpi=150)


def coherence() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    names = (("start", "least-squares\nstart"), ("ls_copy", "copy trained\nfrom the start"),
             ("std_copy", "standard\ncopy"))
    w = 0.38
    for ax, t in zip(axes, ("L", "E")):
        coh = OUT["weights"][t]["coherence"]
        for i, (c, _) in enumerate(names):
            b1 = ax.bar(i - w / 2, coh[c]["weighted"]["kappa"], w, color=COLOUR[c])
            b2 = ax.bar(i + w / 2, coh[c]["frobenius"]["kappa"], w, color="white", lw=1.2, hatch="//",
                        edgecolor=COLOUR[c])
            label_bars(ax, b1)
            label_bars(ax, b2)
        ax.axhline(0.5, color=INK, lw=0.9, ls="--")
        ax.text(3.35, 0.5, "cancels below 0.5", fontsize=7, color=INK, va="top", ha="right")
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.text(3.35, 1, "sign-flip null\nabout 1", fontsize=7, color=MUTED, va="bottom", ha="right")
        ax.set_xticks(range(3), [n for _, n in names], fontsize=8)
        ax.set_yscale("log")
        ax.set_xlim(-0.5, 3.4)
        ax.set_title(TASK[t], fontsize=9)
    axes[0].set_ylabel("coherence of the per-subcomponent changes")
    axes[0].set_ylim(0.2, 12)
    legend(axes[0], "input-weighted (scored)", "Frobenius", "//")
    fig.suptitle("Do the per-subcomponent changes cancel on the failing states only?", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "coherence.png", dpi=150)


def amend1_hold() -> None:
    """Amendment 1, lowercase: every state free and held, in nats, with W0 free and W0 held as reference lines."""
    a1 = json.loads((HERE / "results" / "ls_start_faith_amend1.json").read_text())
    rules = json.loads((HERE / "results" / "predictions_amend1.json").read_text())["config"]
    kl = {**OUT["kl"], **a1["kl"]}
    states = (("W0", "W0", "W0"), ("L.nothing", "nothing", NAME["nothing"]),
              ("L.nothing_start", "nothing_start", NAME["nothing_start"]), ("L.start", "start", NAME["start"]),
              ("L.std_copy", "std_copy", "standard\ncopy"), ("L.ls_copy", "ls_copy", "copy trained\nfrom the start"))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (key, c, _) in enumerate(states):
        label_bars(ax, ax.bar(i - w / 2, sum(kl[key]) / 2, w, color=COLOUR[c]))
        label_bars(ax, ax.bar(i + w / 2, sum(kl[f"{key}|hold"]) / 2, w, color="white", edgecolor=COLOUR[c],
                              hatch="///", lw=1.2))
    w0, w0h = sum(kl["W0"]) / 2, sum(kl["W0|hold"]) / 2
    n = dict(rules["nothing_ref"])["L"]
    bar = (1 + max(rules["floor"], rules["teeth_share"] * (n - 1))) * w0h  # R1: do-nothing held must reach this
    ax.axhline(w0, color=MUTED, lw=0.8, ls="--")
    ax.text(5.62, w0, "W0 free", fontsize=7, color=MUTED, va="bottom", ha="left")
    ax.axhline(w0h, color=INK, lw=0.9, ls=":")
    ax.text(5.62, w0h, "W0 held", fontsize=7, color=INK, va="bottom", ha="left")
    ax.plot([1 - 0.4, 1 + 0.4], [bar, bar], color=INK, ls="--", lw=0.9)
    ax.text(1 + 0.42, bar, f"teeth bar\nfor do-nothing held\n{bar:.2f}", fontsize=7, color=INK, va="center")
    ax.set_xticks(range(len(states)), [s[2] for s in states], fontsize=8)
    ax.set_xlim(-0.5, 6.1)
    ax.set_ylabel("PGD KL, mean of two blocks (nats)")
    legend(ax, "free adversary", "alive masks held at 1", "///")
    ax.set_title(f"{TASK['L']}: under the hold, does any state keep its excess over W0 held?\n"
                 "teeth bar: held do-nothing must keep half its free excess, as a ratio over W0 read the same way",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "amend1_hold.png", dpi=150)


def amend2_delta() -> None:
    """Amendment 2, lowercase: each change on the alive subcomponents (solid) and the same change in Delta (hatched),
    with do-nothing, the fail line and its band of one floor. Ratios over W0 read in the same job."""
    a2 = json.loads((HERE / "results" / "ls_start_faith_amend2.json").read_text())["score"]["ratio"]
    line = json.loads((HERE / "results" / "predictions_amend2.json").read_text())["config"]["fail_line"]
    e = OUT["score"]["L"]["ratio"]
    groups = (("start", "least-squares\nstart (E127)", e["start"], e["nothing_start"]),
              ("ls_copy", "copy trained\nfrom the start", a2["ls_copy"], a2["ls_copy|delta"]),
              ("std_copy", "standard\ncopy", a2["std_copy"], a2["std_copy|delta"]),
              ("lora_copy", "copy of the\nrank-8 LoRA", a2["lora_copy"], a2["lora_copy|delta"]))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.4))
    label_bars(ax, ax.bar(0, e["nothing"], w, color="white", edgecolor=COLOUR["nothing"], hatch="///", lw=1.2))
    for i, (c, _, on, ind) in enumerate(groups, start=1):
        label_bars(ax, ax.bar(i - w / 2, on, w, color=COLOUR[c]))
        label_bars(ax, ax.bar(i + w / 2, ind, w, color="white", edgecolor=COLOUR[c], hatch="///", lw=1.2))
        ax.text(i, max(on, ind) + 0.75, f"excess x{(on - 1) / (ind - 1):.2f}\non subcomponents", fontsize=7,
                color=INK, ha="center")
    ax.axhspan(line - F, line + F, color=MUTED, alpha=0.18, lw=0)
    ax.axhline(line, color=INK, lw=0.9, ls="--")
    ax.text(4.62, line + F, f"fail line {line}\n(band: one floor)", fontsize=7, color=INK, va="bottom", ha="left")
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.text(4.62, 0.95, "W0 = 1", fontsize=7, color=MUTED, va="top", ha="left")
    ax.set_xticks(range(5), ["do-nothing\n(dW in Delta)", *(g[1] for g in groups)], fontsize=8)
    ax.set_xlim(-0.5, 5.2)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("adversarial ratio (PGD KL over W0's)")
    legend(ax, "change on the alive subcomponents", "the same change in Delta", "///")
    ax.set_title(f"{TASK['L']}: does each change read better on the alive subcomponents or in Delta?\n"
                 "the start's placement multiplies its excess; teacher training's divides it", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "amend2_delta.png", dpi=150)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    source()
    hold()
    random_masks()
    coherence()
    amend1_hold()
    amend2_delta()
