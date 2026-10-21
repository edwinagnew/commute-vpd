"""Figures for new_vs_existing: which per-subcomponent numbers separate the two copy tasks (existing mechanism) from
the two random-mapping tasks (new mechanism)?

Usage (repo root): uv run python experiments/new_vs_existing/plot.py [--smoke]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
CLASSES = ("existing", "new")
COLOUR = {"existing": "#1f77b4", "new": "#ff7f0e"}  # fixed: copy tasks blue, random-mapping tasks orange, in every figure
GREY = "#7f7f7f"
# The key-value round's lifted states (experiments/kv_task/results/kv_task.json, 2026-10-19): identity, 32 random pairs.
KV_REFERENCE = {"existing": {"r": 0.802, "e_gate": 0.058, "e_delta": 0.128}, "new": {"r": 0.273, "e_gate": 0.519, "e_delta": 0.184}}


def states(tasks: dict[str, Any], cls: str) -> list[tuple[str, str, dict[str, Any]]]:
    return [(n, s, r) for n, t in tasks.items() if t["cls"] == cls for s, r in t["seeds"].items()]


def dots(ax: Any, tasks: dict[str, Any], get: Any, per_task: bool = False) -> None:
    """One marker per state (or per task) at x = class, filled circles for seed 0 and squares for seed 1."""
    for i, cls in enumerate(CLASSES):
        if per_task:
            vals = [(get(t), "o") for t in tasks.values() if t["cls"] == cls]
        else:
            vals = [(get(r), "o" if s == "0" else "s") for _, s, r in states(tasks, cls)]
        x = i + np.linspace(-0.12, 0.12, len(vals))
        for xi, (v, m) in zip(x, vals):
            ax.plot(xi, v, m, color=COLOUR[cls], ms=8, mec="k", mew=0.4)
    ax.set_xticks([0, 1], ["copy tasks\n(existing)", "random mapping\n(new)"], fontsize=8)
    ax.set_xlim(-0.6, 1.6)


def separation_figure(out: dict[str, Any], suffix: str) -> None:
    tasks = out["tasks"]
    panels = [
        ("Outside share of the update\n(÷ isotropic null; a random update reads 1)", lambda r: r["inexpressible"]["ratio"], False,
         [(1.0, "random update reads 1")]),
        ("Behaviour left in Δ by the least-squares split\n(share of the log-odds gain)", lambda r: r["routes_lift"]["shares"]["e_delta"], False,
         [(KV_REFERENCE["existing"]["e_delta"], "key-value round, identity"), (KV_REFERENCE["new"]["e_delta"], "key-value round, 32 random pairs")]),
        ("Were the moved subcomponents doing the job at W₀?\n(ablation effect of T ÷ CI-matched random sets)", lambda r: r["relevance_W0"]["k_lo"], False,
         [(1.0, "random sets read 1")]),
        ("Re-switching share r\n(gain carried by existing pieces Γ switches on)", lambda r: r["routes_lift"]["shares"]["r"], False,
         [(KV_REFERENCE["existing"]["r"], "key-value round, identity"), (KV_REFERENCE["new"]["r"], "key-value round, 32 random pairs")]),
        ("Masked-off share e_gate\n(gain carried by pieces Γ keeps off)", lambda r: r["routes_lift"]["shares"]["e_gate"], False,
         [(KV_REFERENCE["existing"]["e_gate"], "key-value round, identity"), (KV_REFERENCE["new"]["e_gate"], "key-value round, 32 random pairs")]),
        ("Do two seeds move the same subcomponents?\n(Jaccard of the top-100 sets)", lambda t: t["across_seeds"]["jaccard_T"], True,
         [(float(np.mean([t["across_seeds"]["jaccard_random_T"] for t in tasks.values()])), "two random sets")]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.2))
    for ax, (title, get, per_task, refs) in zip(axes, panels):
        dots(ax, tasks, get, per_task)
        for y, label in refs:
            ax.axhline(y, color=GREY, lw=0.8, ls="--")
            ax.text(1.55, y, label, fontsize=6.5, color=GREY, ha="right", va="bottom")
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].plot([], [], "o", color="w", mec="k", label="seed 0")
    axes[0].plot([], [], "s", color="w", mec="k", label="seed 1")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Which numbers separate the copy tasks from the random-mapping tasks? One marker per fine-tuned state "
                 "(two tasks × two seeds per class); dashed lines are the references", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"separation{suffix}.png", dpi=150)
    print(FIGURES / f"separation{suffix}.png")


def route_figure(out: dict[str, Any], suffix: str) -> None:
    """The research log's Overview figure (2026-10-20): the two route shares that separate the classes beside the one
    update-side number that does not. Current terms only."""
    tasks = out["tasks"]
    panels = [
        ("Outside share of the update\n(÷ isotropic null; a random update reads 1)", lambda r: r["inexpressible"]["ratio"],
         [(1.0, "random update")]),
        ("Re-switching share r\n(gain carried by existing pieces Γ switches on)", lambda r: r["routes_lift"]["shares"]["r"],
         [(KV_REFERENCE["existing"]["r"], "E10 identity"), (KV_REFERENCE["new"]["r"], "E10 32 pairs")]),
        ("Masked-off share e_gate\n(gain carried by pieces Γ keeps off)", lambda r: r["routes_lift"]["shares"]["e_gate"],
         [(KV_REFERENCE["existing"]["e_gate"], "E10 identity"), (KV_REFERENCE["new"]["e_gate"], "E10 32 pairs")]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.9))
    for ax, (title, get, refs) in zip(axes, panels):
        dots(ax, tasks, get)
        for y, label in refs:
            ax.axhline(y, color=GREY, lw=0.8, ls="--")
            ax.text(1.95, y, label, fontsize=7, color=GREY, ha="right", va="bottom")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.6, 2.0)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].plot([], [], "o", color="w", mec="k", label="seed 0")
    axes[0].plot([], [], "s", color="w", mec="k", label="seed 1")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("A behaviour the model has (copy tasks) against one it lacks (random mappings): "
                 "the update looks the same, the route differs", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"route_separation{suffix}.png", dpi=150)
    print(FIGURES / f"route_separation{suffix}.png")

def relevance_figure(out: dict[str, Any], suffix: str) -> None:
    """Per state: the drop in the target's log-odds at the trigger when T is ablated, against the random sets, at W0
    and in the fine-tuned model."""
    tasks = out["tasks"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, key, title in zip(axes, ("relevance_W0", "relevance_ft"),
                              ("At W₀: does ablating the moved subcomponents T lower the target's log-odds?",
                               "In the fine-tuned model: does ablating T remove the trained behaviour?")):
        labels, x = [], 0
        for cls in CLASSES:
            for n, s, r in states(tasks, cls):
                rel = r[key]
                rnd = [q["drop"] for q in rel["random"]]
                ax.plot([x] * len(rnd), rnd, "_", color=GREY, ms=14, mew=1.5, label="CI-matched random sets" if x == 0 else None)
                ax.plot(x, rel["T"]["drop"], "o", color=COLOUR[cls], ms=8, mec="k", mew=0.4,
                        label=f"T, {cls} tasks" if (n, s) == states(tasks, cls)[0][:2] else None)
                labels.append(f"{n}\nseed {s}")
                x += 1
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_ylabel("drop in target log-odds at the trigger (nats)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / f"relevance{suffix}.png", dpi=150)
    print(FIGURES / f"relevance{suffix}.png")


def landing_figure(out: dict[str, Any], suffix: str) -> None:
    """Where the update lands: T's mean causal importance at the trigger at W0 and under the trained activations, the
    lift energy on subcomponents switched off at the trigger, and T's on-rate over Pile against the alive mean."""
    tasks = out["tasks"]
    panels = [("Mean CI of T at the trigger, W₀ activations", lambda r: r["T"]["g0_mean"], [], (0, 1)),
              ("Mean CI of T at the trigger, trained activations", lambda r: r["T"]["g1_mean"], [], (0, 1)),
              ("Lift energy on subcomponents\nswitched off at the trigger (share)", lambda r: r["T"]["energy_share_off_at_trigger"], [], (0, 1)),
              ("How often T is on over Pile at W₀\n(share of positions with CI > 0.5)", lambda r: r["T"]["pile_on_rate_mean"],
               [(out["pile"]["alive_on_rate_mean"], "mean over all alive subcomponents")], (0, 0.2))]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4))
    for ax, (title, get, refs, ylim) in zip(axes, panels):
        dots(ax, tasks, get)
        for y, label in refs:
            ax.axhline(y, color=GREY, lw=0.8, ls="--")
            ax.text(1.55, y, label, fontsize=6.5, color=GREY, ha="right", va="bottom")
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.suptitle("Where does the update land? T = the 100 subcomponents the lift moves most, per fine-tuned state", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"landing{suffix}.png", dpi=150)
    print(FIGURES / f"landing{suffix}.png")


def gamma_figure(out: dict[str, Any], suffix: str) -> None:
    """Gamma's response at the trigger: how many subcomponents it switches on for the trained model's activations, how
    much the two seeds' switched-on sets overlap, and where the switched-on subcomponents sit."""
    tasks = out["tasks"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    dots(axes[0], tasks, lambda r: r["S_on"]["size"])
    axes[0].set_ylim(0, None)
    axes[0].set_title("Subcomponents Γ switches on at the trigger\n(mean CI up by more than 0.3)", fontsize=8.5)
    dots(axes[1], tasks, lambda t: t["across_seeds"]["jaccard_S_on"], per_task=True)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Do two seeds switch on the same set?\n(Jaccard of the switched-on sets)", fontsize=8.5)
    layers = ["0", "1", "2", "3"]
    width = 0.1
    all_states = [(c, n, s, r) for c in CLASSES for n, s, r in states(tasks, c)]
    for j, (cls, n, s, r) in enumerate(all_states):
        counts = {lay: 0 for lay in layers}
        for m, _ in r["S_on"]["members"]:
            counts[m.split(".")[1]] += 1
        axes[2].bar(np.arange(4) + (j - 3.5) * width, [counts[lay] for lay in layers], width, color=COLOUR[cls], edgecolor="k", lw=0.3,
                    label=f"{cls} tasks" if j in (0, 4) else None)
    axes[2].set_xticks(np.arange(4), [f"layer {lay}" for lay in layers], fontsize=8)
    axes[2].set_title("Where the switched-on subcomponents sit\n(one bar per state)", fontsize=8.5)
    axes[2].legend(fontsize=7)
    for ax in axes[:2]:
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.suptitle("How does the frozen Γ respond to the trained model at the trigger?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"gamma{suffix}.png", dpi=150)
    print(FIGURES / f"gamma{suffix}.png")


def main() -> None:
    suffix = "_smoke" if "--smoke" in sys.argv else ""
    out = json.loads((RESULTS / f"new_vs_existing{suffix}.json").read_text())
    FIGURES.mkdir(exist_ok=True)
    separation_figure(out, suffix)
    route_figure(out, suffix)
    relevance_figure(out, suffix)
    landing_figure(out, suffix)
    gamma_figure(out, suffix)


if __name__ == "__main__":
    main()
