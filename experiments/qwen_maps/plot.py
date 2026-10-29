"""Figures for E44 (Q0 + Q1 on Qwen3-1.7B): one figure per claim, each with its reference on the same axes.

Reads results/{C,S,D,K}.json, C_2026-10-28_volume.json and predictions.json only (no model).
Usage (repo root): uv run python experiments/qwen_maps/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"
STATES = {  # key: (name in words, colour); the same colour for the same state in every figure
    "do_nothing": ("do-nothing:\nwhole update in Δ", "#c0392b"),
    "lower": ("split onto the rarely\nswitched-on half", "#2a78d6"),
    "upper": ("split onto the often\nswitched-on half", "#7b4fc9"),
    "random": ("random change of\nthe same size", "0.35"),
    "ls": ("least-squares split", "#1a9850"),
    "inspan": ("least squares,\nin-span part only", "#66bd63"),
    "adjoint": ("scaled adjoint split", "#e69f00"),
}
STORE_ERR_MAX = 1e-3  # run.py's stop line for the stored pieces' precision error
LINE = 1.5  # the pass test's line for ratios to W0


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=150)
    plt.close(fig)


def reproduction() -> None:
    """W0's readings in each pod against the values filed from E40, as a percent difference."""
    filed = load("predictions")["rules"]["control_W0"]
    refs = {"masked KL,\nmaths rows": filed["ci"]["maths"], "masked KL,\nweb rows": filed["ci"]["web"]}
    refs |= {f"adversary {i + 1}\nreplayed": v for i, v in enumerate(filed["replay"].values())}
    refs |= {f"KL to start,\nstep {k.split('_')[0]}": v for k, v in filed["kl_to_start"].items()}

    def read(c: dict) -> list[float]:
        vals = [c["ci"]["maths"], c["ci"]["web"], *(c["replay"][k] for k in filed["replay"])]
        vals += [c["kl_to_start"][k] for k in filed["kl_to_start"]]
        return [100 * (v / r - 1) for v, r in zip(vals, refs.values(), strict=True)]

    x = range(len(refs))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(-100 * filed["tol"], 100 * filed["tol"], color="0.9", label="filed tolerance, ±1%")
    ax.axhline(0, color="0.55", lw=1, label="E40's value")
    ax.plot(x, read(load("C_2026-10-28_volume")), "s", mfc="none", color="0.35", ms=8, label="first pod")
    ax.plot(x, read(load("C")), "o", color="#2a78d6", label="scored pod")
    ax.set_xticks(list(x), list(refs), fontsize=8)
    ax.set_ylabel("difference from E40's value (%)")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Does the unchanged decomposition read as it did in E40?")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "reproduction")


def precision() -> None:
    """Per matrix, how far the stored pieces' applied change is from the float64 one, relative to the update."""
    S = load("S")
    keys = ["ls", "lower", "upper"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(keys):
        for j, (t, mk) in enumerate((("1", "^"), ("10", "o"))):
            errs = [max(r["stored_error"][k], 1e-12) for r in S[t]["per_site"].values()]
            xs = [i + (j - 0.5) * 0.3] * len(errs)
            ax.plot(xs, errs, mk, color=STATES[k][1], alpha=0.35, ms=4,
                    label=f"step {t}: one point per matrix" if i == 0 else None)
    ax.axhline(STORE_ERR_MAX, color="#c0392b", ls="--", label="stop line: 1e-3 of the update")
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)), [STATES[k][0] for k in keys], fontsize=8)
    ax.set_ylabel("precision error of the applied split\n(fraction of the update's size)")
    ax.set_title("Are the stored split pieces as precise as float64?")
    ax.set_ylim(5e-10, 1e-2)
    leg = ax.legend(fontsize=8, loc="upper right")
    for h in leg.legend_handles[:2]:
        h.set_color("0.3")
    save(fig, "precision")


def shares() -> None:
    """Share of the update left in Δ by each state, steps 1 and 10."""
    S = load("S")
    keys = ["do_nothing", "lower", "upper", "ls", "inspan", "adjoint", "random"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, k in enumerate(keys):
        ax.bar(i - 0.18, S["1"]["share"][k], 0.36, color=STATES[k][1], alpha=0.45, label="step 1" if i == 0 else None)
        ax.bar(i + 0.18, S["10"]["share"][k], 0.36, color=STATES[k][1], label="step 10" if i == 0 else None)
        ax.text(i + 0.18, S["10"]["share"][k] + 0.02, f"{S['10']['share'][k]:.3g}", ha="center", fontsize=7)
    lo, up = S["10"]["share"]["lower"], S["10"]["share"]["upper"]
    ax.annotate("", (1.18, lo), (2.18, up), arrowprops={"arrowstyle": "<->", "color": "0.2"})
    ax.text(1.68, lo + 0.08, f"gap {lo - up:.2f};\nfiled limit 0.1", ha="center", fontsize=8)
    ax.axhline(1, color="0.55", lw=1, ls=":", label="all of the update")
    ax.set_xticks(range(len(keys)), [STATES[k][0] for k in keys], fontsize=7)
    ax.set_ylabel("share of the update left in Δ\n(input-weighted size)")
    ax.set_ylim(0, 1.15)
    ax.set_title("How much of the update does each state leave outside the subcomponents?")
    ax.legend(fontsize=8)
    save(fig, "shares")


def kept() -> None:
    """KL from the fine-tuned model to each control's masked model, against K(D) and K0."""
    C, D, K = load("C"), load("D"), load("K")
    keys = ["do_nothing", "lower", "upper", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, t in zip(axes, ("1", "10"), strict=True):
        vals = [D[t]["do_nothing"]["maths"]["K"], *(K[t][k]["maths"]["K"] for k in keys[1:])]
        ax.bar(range(len(keys)), vals, color=[STATES[k][1] for k in keys])
        for i, v in enumerate(vals):
            ax.text(i, v * 1.08, f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(D[t]["KD"], color="#c0392b", ls="--", label="unchanged decomposition:\nupdate dropped")
        ax.axhline(C["ci"]["maths"], color="0.4", ls=":", label="original model vs its own\nmasked model")
        ax.set_xticks(range(len(keys)), [STATES[k][0] for k in keys], fontsize=7)
        ax.set_title(f"fine-tuning step {t}")
        ax.set_yscale("log")
    axes[0].set_ylabel("KL from the fine-tuned model\nto the masked model, maths rows (nats)")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Does a state that leaves the update out of its subcomponents lose the update?")
    save(fig, "kept")


def ratios() -> None:
    """The controls' ratios to W0 at step 10, against the pass line."""
    D, K = load("D"), load("K")
    reads = {"r1m": "masked KL,\nmaths", "r1w": "masked KL,\nweb", "r3m": "adversary,\nmaths", "r3w": "adversary,\nweb"}
    keys = ["do_nothing", "lower", "upper", "random"]
    recs = {"do_nothing": D["10"]["do_nothing"]} | {k: K["10"][k] for k in keys[1:]}
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    for i, k in enumerate(keys):
        xs = [j + (i - 1.5) * w for j in range(len(reads))]
        ax.bar(xs, [recs[k][r] for r in reads], w, color=STATES[k][1], label=STATES[k][0].replace("\n", " "))
    ax.axhline(LINE, color="#c0392b", ls="--", label="pass line, 1.5")
    ax.axhline(1, color="0.55", lw=1, label="unchanged decomposition, 1")
    ax.set_xticks(range(len(reads)), list(reads.values()), fontsize=8)
    ax.set_ylabel("ratio to the unchanged decomposition\non the same rows, step 10")
    ax.set_ylim(0, 1.9)
    ax.set_title("Does a random change of the update's size push any ratio over the line?")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    save(fig, "ratios")


if __name__ == "__main__":
    reproduction()
    precision()
    shares()
    kept()
    ratios()
