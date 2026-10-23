"""Figure for kv_task Amendment 1: does the decomposition's Pile adversarial ratio follow the state's damage elsewhere
or its size of change? Every scored state, coloured by task, with the pass and fail lines and W0's seed band.

Reads results/kv_task_amend1.json (and E10's ident32 step-30 reading from results/kv_task.json) only.
Usage (repo root): uv run python experiments/kv_task/plot_amend1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
COLOURS = {"rand32": "#2a78d6", "ident32": "#eb6834"}
NAMES = {"rand32": "32 random pairs (cipher)", "ident32": "32 copy pairs (identity control)"}


def load_states() -> tuple[list[dict], dict]:
    amend = json.loads((RESULTS / "kv_task_amend1.json").read_text())
    states = []
    for shard in amend["shards"]:
        for s in shard["states"].values():
            states.append({"task": s["task"], "step": s["step"], "size": s["size"],
                           "damage": s["heldout"]["kl_other"], "pgd": s["pile"]["pgd_kl"],
                           "verdict": s["verdict"], "e10": False})
    e10 = json.loads((RESULTS / "kv_task.json").read_text())["tasks"]["ident32"]["pullback"]
    at30 = next(c for c in e10["curve"] if c["step"] == 30)
    states.append({"task": "ident32", "step": 30, "size": at30["rel_norm_max"], "damage": at30["heldout_kl_other"],
                   "pgd": e10["eval"]["pile"]["ratios"]["pgd_kl"], "verdict": "pass", "e10": True})
    return states, amend["shards"][0]["W0"]


def main() -> None:
    states, w0 = load_states()
    f = w0["pgd_floor"]
    seeds = [w0["pile"][k]["pgd_kl"] / w0["pile"]["0"]["pgd_kl"] for k in ("0", "1", "2")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, key, label in ((axes[0], "damage", "damage elsewhere: held-out KL to W0 off the key (nats)"),
                           (axes[1], "size", "size of change: max over matrices of |dW| / |W0|")):
        ax.axhspan(min(seeds), max(seeds), color="0.85", label="W0 under three adversary seeds")
        ax.axhline(1.5 / (1 + f), color="0.3", lw=1, ls="--", label=f"pass line {1.5 / (1 + f):.2f}")
        ax.axhline(1.5 * (1 + f), color="0.3", lw=1, ls=":", label=f"fail line {1.5 * (1 + f):.2f}")
        for task, colour in COLOURS.items():
            pts = [s for s in states if s["task"] == task]
            ax.scatter([s[key] for s in pts if not s["e10"]], [s["pgd"] for s in pts if not s["e10"]], s=60,
                       color=colour, edgecolor="white", linewidth=1.5, zorder=3, label=NAMES[task])
            old = [s for s in pts if s["e10"]]
            ax.scatter([s[key] for s in old], [s["pgd"] for s in old], s=60, facecolor="none", edgecolor=colour,
                       linewidth=1.5, zorder=3, label="identity at step 30, read in E10" if old else None)
            for s in pts:
                ax.annotate(f"step {s['step']}", (s[key], s["pgd"]), xytext=(6, -10 if s["step"] == 300 else 4), textcoords="offset points",
                            fontsize=7, color="0.25")
        ax.set_xlabel(label, fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Pile adversarial ratio to W0 (1 = no change)", fontsize=9)
    axes[0].set_title("against damage elsewhere", fontsize=9)
    axes[1].set_title("against size of change", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("The cipher state fails only where its damage elsewhere is largest, not where it is largest in size",
                 fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "amend1_ratio.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
