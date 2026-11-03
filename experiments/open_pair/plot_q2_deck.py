"""The Q2 slide's figures (new against existing), from saved results only, with no model.

  q2_routes.png        E12: outside share, r and the masked-off share, copy tasks against random maps
  q2_every_family.png  E60: r and steering on three task families, built to reuse against built to be new
  q2_r_vs_prior.png    E58 and E62: r against the original model's prior of the trained answer
  q2_overlap.png       E67: the overlap share and how far Γ's values move

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/open_pair/plot_q2_deck.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_amend7 import read

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


def headline(fig: Any, title: str, y: float = 1.12) -> None:
    fig.suptitle(title, x=0.01, y=y, ha="left", fontsize=17, fontweight="bold")


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
        ("Update outside the pieces", {"existing": p1["existing"], "new": p1["new"]}),
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
        ax.tick_params(labelleft=True)
    axes[0].set_ylabel("Ratio to a random update")
    axes[1].set_ylabel("Share of the fine-tune's gain")
    axes[2].set_ylabel("Share of the fine-tune's gain")
    fig.subplots_adjust(wspace=0.32)
    headline(fig, "Copying and random maps look alike in the update but take different routes")
    save(fig, "q2_routes.png")


def every_family() -> None:
    runs = json.loads((ROOT / "experiments/regating_truth_table/results/table.json").read_text())["runs"]
    families = (("Copy vs\nrandom maps", ("e46/ident32",), ("e46/rand32",)),
                ("Lowercase\nvs ciphers", ("e53/lowercase",), ("e53/cipher", "e53/broad_cipher")),
                ("Half-known\nvs cipher", ("e54/elicit",), ("e54/cipher",)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, key, title in zip(axes, ("r", "S"), ("r: gain from old pieces\nthat Γ switches on", "Steering: gain from one mean vector,\nno decomposition needed")):
        for x, (_, reuse, new_side) in enumerate(families):
            for prefixes, color, centre in ((reuse, BLUE, x - 0.17), (new_side, ORANGE, x + 0.17)):
                values = [v[key]["value"] for k, v in runs.items() if k.rsplit("_s", 1)[0].removesuffix("b") in prefixes or k.rsplit("_s", 1)[0] in prefixes]
                spread = [centre + 0.05 * (i - (len(values) - 1) / 2) for i in range(len(values))]
                ax.scatter(spread, values, s=110, color=color, edgecolor="white", lw=1, zorder=3)
        ax.set_xticks(range(3), [f[0] for f in families])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0, 1.05)
        clean(ax, title, "y")
        ax.tick_params(labelleft=True)
        ax.set_ylabel("Share of the fine-tune's gain")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=BLUE, label="Built on something the model already does"),
               plt.Line2D([], [], marker="o", ls="", ms=10, color=ORANGE, label="Built to be new")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.07))
    headline(fig, "r keeps a wide gap on every family we built; steering does so only on copy")
    save(fig, "q2_every_family.png")


LABEL = {"rank1": "rank-1", "elicit": "half-known", "copy": "copy", "prior_cipher": "prior-matched cipher", "shuffle": "shuffled half-known",
         "cipher": "cipher", "class_shuffled": "shuffled twin", "class_map": "class rule"}
OFFSET = {"rank1": (-11, 2, "right"), "elicit": (11, -9, "left"), "copy": (-11, 4, "right"), "prior_cipher": (9, -14, "left"), "shuffle": (9, 5, "left"),
          "cipher": (11, -8, "left"), "class_shuffled": (-4, 16, "left"), "class_map": (7, -15, "left")}
KNOWN = ("rank1", "elicit", "copy")


def r_vs_prior() -> None:
    prior = read("amend7")["prior_beside_r"]
    run = json.loads((ROOT / "experiments/induction_cue/results/run.json").read_text())
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for m, v in prior.items():
        x, y = v["w0_geo_p_trained"], mean(v["r30"])
        ax.scatter([x], [y], s=115, color=BLUE if m in KNOWN else ORANGE, edgecolor="white", lw=1, zorder=3)
        off = OFFSET[m]
        ax.annotate(LABEL[m], (x, y), xytext=off[:2], textcoords="offset points", ha=off[2], va="center", fontsize=12.5)
    for kind, key, color, off in (("copy", "prior_copy", BLUE, (-12, 0, "right")), ("lookup", "prior_lookup", ORANGE, (12, -2, "left"))):
        x, y = run["stage_b"][key], mean([run["states"][f"{kind}_s{s}"]["r"] for s in range(3)])
        ax.scatter([x], [y], s=125, marker="D", color=color, edgecolor="white", lw=1, zorder=3)
        ax.annotate(f"{kind} at a new marker", (x, y), xytext=off[:2], textcoords="offset points", ha=off[2], va="center", fontsize=12.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e-1)
    ax.set_xticks([1e-7, 1e-5, 1e-3, 1e-1])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("r: share of the gain from Γ's switching alone")
    ax.set_xlabel("Original model's probability of the trained answer, before any fine-tune")
    clean(ax, "", "y")
    headline(fig, "Caveat: the base model's logits correlate with r", 0.96)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=10, color=BLUE, label="Built on something the model already does"),
               plt.Line2D([], [], marker="o", ls="", ms=10, color=ORANGE, label="Built to be new"),
               plt.Line2D([], [], marker="D", ls="", ms=9, color=GREY, label="Diamond: taught at a new marker")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
    save(fig, "q2_r_vs_prior.png")


def overlap() -> None:
    run = json.loads((ROOT / "experiments/weight_side_read/results/run.json").read_text())
    groups = (("Copy\nfine-tunes", ("state_ident32_s0", "state_ident32_s1"), BLUE), ("Learned\nnothing", ("state_code_step10", "state_code_step100"), ORANGE),
              ("Random\nupdate", ("state_gaussian",), GREY))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.3))
    for x, (_, keys, color) in enumerate(groups):
        offs = [-0.07, 0.07] if len(keys) == 2 else [0.0]
        left.scatter([x + o for o in offs], [run[k]["S_R"] for k in keys], s=110, color=color, edgecolor="white", lw=1, zorder=3, clip_on=False)
        right.scatter([x + o for o in offs], [run[k]["mean_abs_dg"] * 1e4 for k in keys], s=110, color=color, edgecolor="white", lw=1, zorder=3)
    for ax in (left, right):
        ax.set_xticks(range(3), [g[0] for g in groups])
        ax.set_xlim(-0.5, 2.5)
    left.set_ylim(0, 1.05)
    right.set_ylim(0, 9)
    clean(left, "Does Γ's switching do what the update does?", "y")
    clean(right, "How far Γ's values move", "y")
    left.set_ylabel("Overlap (1 = fully)")
    right.set_ylabel("Mean change per piece (×10⁻⁴)")
    fig.subplots_adjust(wspace=0.25)
    headline(fig, "Overlap is low, and Γ moves even on a fine-tune that learned nothing", 1.05)
    save(fig, "q2_overlap.png")


def main() -> None:
    routes()
    every_family()
    r_vs_prior()
    overlap()


if __name__ == "__main__":
    main()
