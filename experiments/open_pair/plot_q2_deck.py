"""The Q2 slide's figures (new against existing), from saved results only, with no model.

  q2_routes.png          E12: outside share, r and the masked-off share, copy tasks against random maps
  q2_three_readings.png  method table: W0's prior, r and steering on the same eight pairs
  q2_r_prior_carry.png   E58 and E62: r against the prior, and E58's r against carry
  q2_copying_head.png    E62: the prior, the gain lost with the copying head off, and r
  q2_overlap.png         E67: the overlap share and how far Γ's values move

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/open_pair/plot_q2_deck.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_amend7 import carry30, read

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "experiments/open_pair/figures"
BLUE, ORANGE, PURPLE, GREY = "#2a78d6", "#e07a00", "#7a4fd0", "#8a8983"
INK, GRID = "#3f3e3b", "#e4e3df"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def clean(ax: Any, title: str, grid: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=14, pad=10)


def mean(v: list[float]) -> float:
    return sum(v) / len(v)


def save(fig: Any, name: str) -> None:
    fig.savefig(FIG / name, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def routes() -> None:
    tasks = json.loads((ROOT / "experiments/new_vs_existing/results/new_vs_existing.json").read_text())
    p1 = tasks["score"]["P1"]
    states = [(t["cls"], s["routes_lift"]["shares"]) for t in tasks["tasks"].values() for s in t["seeds"].values()]
    panels = [
        ("Update outside the pieces\n(a random update reads 1)", {"existing": p1["existing"], "new": p1["new"]}),
        ("Gain from old pieces\nthat Γ switches on (r)", {c: [s["r"] for k, s in states if k == c] for c in ("existing", "new")}),
        ("Gain from pieces\nthat Γ keeps off", {c: [s["e_gate"] for k, s in states if k == c] for c in ("existing", "new")}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    for ax, (title, values) in zip(axes, panels):
        for x, (cls, color) in enumerate((("existing", BLUE), ("new", ORANGE))):
            offs = [-0.09, -0.03, 0.03, 0.09]
            ax.scatter([x + o for o in offs], values[cls], s=110, color=color, edgecolor="white", lw=1, zorder=3)
        ax.set_xticks([0, 1], ["Copy tasks", "Random maps"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 1.05)
        clean(ax, title, "y")
    save(fig, "q2_routes.png")


ROWS = [
    ("Answers far apart in prior", None),
    ("Rank-1 vs cipher", "e58_rank1_96"),
    ("Half-known vs cipher", "e54_matched"),
    ("Half-known vs its shuffle", "e58_elicit_shuffle"),
    ("Answers close in prior", None),
    ("Copy vs prior-matched cipher", "e58_copy_prior_cipher"),
    ("Copy at new marker vs lookup", "e62_copy_lookup"),
    ("Class rule vs its shuffled twin", "e58_class_twin"),
    ("Prior not read", None),
    ("Copy vs random maps", "e46_copy_random"),
    ("Lowercase vs ciphers", "e53_lowercase_cipher"),
]


def pair(ax: Any, y: float, first: float, second: float) -> None:
    ax.plot([first, second], [y, y], color=GREY, lw=2.2, zorder=2)
    ax.scatter([second], [y], s=95, facecolor="white", edgecolor=ORANGE, lw=2.2, zorder=3)
    ax.scatter([first], [y], s=95, color=BLUE, edgecolor="white", lw=1, zorder=4)


def three_readings() -> None:
    table = json.loads((ROOT / "experiments/method_table/results/table.json").read_text())
    rows = {r["id"]: r["cells"] for r in table["rows"]}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.3), sharey=True, gridspec_kw={"width_ratios": [1.05, 1, 1]})
    labels, ticks = [], []
    for i, (name, row) in enumerate(ROWS):
        y = len(ROWS) - i
        if row is None:
            axes[0].text(-0.02, y - 0.1, name, transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontweight="bold", fontsize=12.5)
            continue
        labels.append(name)
        ticks.append(y)
        for ax, method, log in zip(axes, ("prior", "r", "steering"), (True, False, False)):
            reading = rows[row][method]["reading"]
            if not reading.get("reuse") or not reading.get("new"):
                x = 3e-5 if log else 0.5
                ax.text(x, y, "not read", color=GREY, ha="center", va="center", fontsize=12)
                continue
            a, b = mean(reading["reuse"]), mean(reading["new"])
            pair(ax, y, math.exp(a) if log else a, math.exp(b) if log else b)
    axes[0].set_yticks(ticks, labels)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-7, 1e-1)
    axes[0].set_xticks([1e-7, 1e-5, 1e-3, 1e-1])
    for ax in axes[1:]:
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    for ax, title in zip(axes, ("Original model's probability\nof the trained answer", "r: share of the gain\nfrom Γ's switching alone",
                                "Steering: share of the gain\nfrom one mean vector")):
        clean(ax, title)
    axes[0].set_ylim(0.4, len(ROWS) + 0.6)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=BLUE, label="First map named"),
               plt.Line2D([], [], marker="o", ls="", ms=10, markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=2, label="Second map named")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.05))
    save(fig, "q2_three_readings.png")


LABEL = {"rank1": "rank-1", "elicit": "half-known", "copy": "copy", "prior_cipher": "prior-matched cipher", "shuffle": "shuffled half-known",
         "cipher": "cipher", "class_shuffled": "shuffled twin", "class_map": "class rule"}
OFFSET_PRIOR = {"rank1": (-10, 2, "right"), "elicit": (10, -10, "left"), "copy": (-10, 4, "right"), "prior_cipher": (8, -14, "left"), "shuffle": (8, 4, "left"),
                "cipher": (10, -9, "left"), "class_shuffled": (-4, 16, "left"), "class_map": (6, -16, "left")}
OFFSET_CARRY = {"rank1": (10, 0, "left"), "elicit": (10, 2, "left"), "copy": (-10, 6, "right"), "prior_cipher": (8, 6, "left"), "shuffle": (10, 6, "left"),
                "cipher": (-10, -2, "right"), "class_shuffled": (8, 6, "left"), "class_map": (10, 0, "left")}


def r_prior_carry() -> None:
    a7 = read("amend7")
    prior, carry = a7["prior_beside_r"], carry30(a7)
    run = json.loads((ROOT / "experiments/induction_cue/results/run.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for m, v in prior.items():
        y = mean(v["r30"])
        left.scatter([v["w0_geo_p_trained"]], [y], s=100, color=BLUE, edgecolor="white", lw=1, zorder=3)
        right.scatter([mean(carry[m])], [y], s=100, color=BLUE, edgecolor="white", lw=1, zorder=3)
        for ax, x, off in ((left, v["w0_geo_p_trained"], OFFSET_PRIOR[m]), (right, mean(carry[m]), OFFSET_CARRY[m])):
            ax.annotate(LABEL[m], (x, y), xytext=off[:2], textcoords="offset points", ha=off[2], va="center", fontsize=12)
    for name, key, off in (("copy at new marker", "prior_copy", (-10, 0, "right")), ("lookup at new marker", "prior_lookup", (10, -2, "left"))):
        kind = name.split()[0]
        y = mean([run["states"][f"{kind}_s{s}"]["r"] for s in range(3)])
        x = run["stage_b"][key]
        left.scatter([x], [y], s=110, marker="D", color=PURPLE, edgecolor="white", lw=1, zorder=3)
        left.annotate(name, (x, y), xytext=off[:2], textcoords="offset points", ha=off[2], va="center", fontsize=12)
    left.set_xscale("log")
    left.set_xlim(1e-7, 1e-1)
    left.set_xticks([1e-7, 1e-5, 1e-3, 1e-1])
    left.set_ylim(0, 1.2)
    left.set_ylabel("r, at step 30")
    left.set_xlabel("Original model's probability of the trained answer")
    right.set_xlim(-0.3, 1)
    right.set_xlabel("Carry: gain at unseen keys ÷ gain at trained keys")
    clean(left, "r against how likely the answer already was", "both")
    clean(right, "r against how far the map carries to unseen keys", "both")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=BLUE, label="Eight key maps, 96 trained keys"),
               plt.Line2D([], [], marker="D", ls="", ms=9, color=PURPLE, label="Two maps taught at a new marker (carry not read)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.07))
    save(fig, "q2_r_prior_carry.png")


def copying_head() -> None:
    run = json.loads((ROOT / "experiments/induction_cue/results/run.json").read_text())
    st, b = run["states"], run["stage_b"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.1), sharey=True)
    for y, (kind, color, key) in enumerate((("lookup", ORANGE, "prior_lookup"), ("copy", BLUE, "prior_copy"))):
        axes[0].scatter([b[key]], [y], s=120, color=color, edgecolor="white", lw=1, zorder=3)
        for ax, field in ((axes[1], "share"), (axes[2], "r")):
            ax.scatter([st[f"{kind}_s{s}"][field] for s in range(3)], [y] * 3, s=120, color=color, edgecolor="white", lw=1, alpha=0.85, zorder=3)
    axes[0].set_yticks([0, 1], ["Lookup at new marker", "Copy at new marker"])
    axes[0].set_ylim(-0.6, 1.6)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-7, 1e-1)
    axes[0].set_xticks([1e-7, 1e-5, 1e-3, 1e-1])
    axes[1].set_xlim(-0.1, 1.2)
    axes[2].set_xlim(-0.1, 1.2)
    for ax, title in zip(axes, ("Original model's probability\nof the trained answer", "Gain lost with the model's\ncopying head switched off", "r: share of the gain\nfrom Γ's switching alone")):
        clean(ax, title)
    save(fig, "q2_copying_head.png")


def overlap() -> None:
    run = json.loads((ROOT / "experiments/weight_side_read/results/run.json").read_text())
    states = [("Copy fine-tune, seed 0", "state_ident32_s0", BLUE), ("Copy fine-tune, seed 1", "state_ident32_s1", BLUE),
              ("Learned nothing, step 10", "state_code_step10", ORANGE), ("Learned nothing, step 100", "state_code_step100", ORANGE),
              ("Random update, same size", "state_gaussian", GREY)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (_, key, color) in enumerate(states):
        y = len(states) - 1 - i
        s = run[key]
        left.barh([y], [s["S_R"]], height=0.55, color=color, zorder=2)
        left.scatter([m["S_R"] for m in s["per_matrix"].values()], [y] * 24, s=20, facecolor="white", edgecolor=INK, lw=0.7, alpha=0.75, zorder=3)
        left.text(0.42, y, f"{s['S_R']:.3f}", va="center", ha="left", fontsize=13)
        right.barh([y], [s["mean_abs_dg"] * 1e4], height=0.55, color=color, zorder=2)
    left.set_yticks(range(len(states)), [s[0] for s in reversed(states)])
    left.set_xlim(-0.12, 1)
    left.set_xlabel("Overlap (0 orthogonal, 1 matches)\nBar and number: all matrices pooled. Dots: each of 24 matrices.")
    right.set_xlabel("Mean change per piece and position\n(×10⁻⁴)")
    right.set_xlim(0, 9)
    clean(left, "Does Γ's switching line up with what the update does?")
    clean(right, "How far Γ's values move")
    save(fig, "q2_overlap.png")


def main() -> None:
    routes()
    three_readings()
    r_prior_carry()
    copying_head()
    overlap()


if __name__ == "__main__":
    main()
