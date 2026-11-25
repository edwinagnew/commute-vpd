"""The research log's four figures for E15, redrawn from saved results with plain labels (no model, no new runs).

  log_control.png   does retraining Γ leave the unchanged model alone? The unchanged model's adversarial ratio at each
                    retraining checkpoint under both recipes, with the no-adversary checks of the first recipe and the
                    pass band. Reads results/gamma_refit.json (refits.w0_P, W0.pgd_floor) and
                    results/gamma_refit_v2.json (partial.w0_P, stopped at step 300).
  log_split.png     does the split state reach the band, and is that readable? The least-squares split state's
                    adversarial ratio under the first recipe, the unchanged model's under the same recipe on the same
                    axes. Reads results/gamma_refit.json (refits.ls_P, refits.w0_P).
  log_sparsity.png  how did Γ get there? VPD's sparsity loss relative to the released Γ on the unchanged model, for the
                    split state and the unchanged model. Reads results/gamma_refit.json (refits.*.checkpoints.*.sparsity_ratio).
  log_stage0.png    before any retraining: are the pieces the adversary turns down more important at the updated model?
                    Per state, the adversary's joint damage relative to the unchanged model's beside the median
                    single-piece ablation ratio of each group of 48. Reads results/gamma_refit.json (stage0).

Usage (repo root):  python experiments/gamma_refit/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from plot import FIGURES, GREY, HERE, step_axis, steps_of, xpos

ORANGE = "#eb6834"
BLUE = "#2a78d6"
INK = "#333333"


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / "results" / name).read_text())


def series(refit: dict[str, Any], key: str) -> tuple[list[int], list[float]]:
    steps = steps_of(refit)
    return steps, [refit["checkpoints"][str(s)]["pile"][key] for s in steps]


def band(ax: Any, out: dict[str, Any], steps: list[int]) -> None:
    top = 1 + 3 * out["W0"]["pgd_floor"]
    ax.axhspan(1, top, color=GREY, alpha=0.25, lw=0)
    ax.text(xpos(steps)[-1] * 1.15, (1 + top) / 2, f"pass band\n(1 to {top:.2f})", fontsize=7.5, color=INK,
            ha="left", va="center")


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(FIGURES / name)


def label_points(ax: Any, xs: Any, ys: list[float], colour: str, dy: int = 7) -> None:
    for x, y in zip(xs, ys, strict=True):
        ax.annotate(f"{y:.2f}" if y < 2 else f"{y:.1f}" if y < 10 else f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=7.5, color=colour)


def fig_control(out: dict[str, Any], v2: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    steps = steps_of(out["refits"]["w0_P"])
    s1, y1 = series(out["refits"]["w0_P"], "pgd_kl")
    s2, y2 = series(v2["partial"]["w0_P"], "pgd_kl")
    _, st = series(out["refits"]["w0_P"], "stochastic_kl")
    ax.plot(xpos(s1), y1, "-o", color=BLUE, lw=2.2, ms=5, label="first recipe: adversarial check")
    label_points(ax, xpos(s1), y1, BLUE)
    ax.plot(xpos(s2), y2, "--s", color=BLUE, lw=1.6, ms=5, alpha=0.7,
            label="second recipe (adversary warmed up, rate 10× lower; stopped at step 300)")
    label_points(ax, xpos(s2)[1:], y2[1:], BLUE, dy=-13)
    ax.plot(xpos(s1), st, ":", color=GREY, lw=1.4, marker="o", ms=3, label="first recipe: random-mask check, no adversary")
    band(ax, out, steps)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 12)
    ax.set_yticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    ax.set_ylabel("adversarial ratio\n(damage ÷ the released decomposition's)")
    step_axis(ax, steps)
    ax.set_xlim(xpos(steps)[0] * 0.6, xpos(steps)[-1] * 3.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.set_title("Retraining Γ on the unchanged model takes it out of its own pass band, under both recipes", fontsize=10)
    save(fig, "log_control.png")


def fig_split(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    steps = steps_of(out["refits"]["ls_P"])
    s, ls = series(out["refits"]["ls_P"], "pgd_kl")
    _, w0 = series(out["refits"]["w0_P"], "pgd_kl")
    ax.plot(xpos(s), ls, "-o", color=ORANGE, lw=2.2, ms=5, label="least-squares split of the fine-tune")
    label_points(ax, xpos(s)[:-1], ls[:-1], ORANGE)
    ax.annotate(f"{ls[-1]:.2f}", (xpos(s)[-1], ls[-1]), textcoords="offset points", xytext=(-16, -12), ha="center",
                fontsize=7.5, color=ORANGE)
    ax.plot(xpos(s), w0, "-o", color=BLUE, lw=1.4, ms=4, alpha=0.8, label="unchanged model, same retraining")
    label_points(ax, xpos(s), w0, BLUE, dy=-13)
    band(ax, out, steps)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 40)
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_ylabel("adversarial ratio\n(damage ÷ the released decomposition's)")
    step_axis(ax, steps)
    ax.set_xlim(xpos(steps)[0] * 0.6, xpos(steps)[-1] * 3.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.set_title("The split reaches the band at step 3,000, but the same retraining moves the unchanged model",
                 fontsize=10)
    save(fig, "log_split.png")


def fig_sparsity(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    steps = steps_of(out["refits"]["ls_P"])
    for tag, lab, colour, dy in (("ls_P", "least-squares split of the fine-tune", ORANGE, 7),
                                 ("w0_P", "unchanged model, same retraining", BLUE, -13)):
        r = out["refits"][tag]
        ys = [r["checkpoints"][str(s)]["sparsity_ratio"] for s in steps]
        ax.plot(xpos(steps), ys, "-o", color=colour, lw=2.0, ms=5, label=lab)
        label_points(ax, xpos(steps)[-1:], ys[-1:], colour, dy=dy)
    ax.axhline(1.0, color=GREY, lw=0.8)
    ax.axhline(1.05, color="#b03a2e", lw=0.8, ls=":")
    ax.text(xpos(steps)[0], 1.06, "1.05: the plan's threshold for Γ keeping more pieces on", fontsize=7.5,
            color="#b03a2e", va="bottom")
    ax.set_ylim(0.85, 1.75)
    ax.set_ylabel("VPD sparsity loss\n÷ the released decomposition's")
    step_axis(ax, steps)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 0.88), frameon=False)
    ax.set_title("As the split entered the band, its sparsity loss rose 60%; the unchanged model's, 9%", fontsize=10)
    save(fig, "log_sparsity.png")


def fig_stage0(out: dict[str, Any]) -> None:
    s0 = out["stage0"]
    states = (("nothing", "do-nothing\n(whole change in Δ)"), ("ls", "least-squares split"), ("ct", "component training"))
    bars = (("joint", "the adversary's mask, all pieces at once", INK),
            ("down_changed", "one piece at a time: turned down, most changed", ORANGE),
            ("up_changed", "one piece at a time: left on, most changed", BLUE),
            ("down_random", "one piece at a time: turned down, random", GREY))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, (name, _) in enumerate(states):
        for b, (key, lab, colour) in enumerate(bars):
            v = s0[name]["pgd_kl"] / s0["w0"]["pgd_kl"] if key == "joint" else s0[name]["groups"][key]["summary"]["median_ratio"]
            x = i + (b - 1.5) * w
            ax.bar(x, v, w * 0.9, color=colour, label=lab if i == 0 else None)
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, fontweight="bold" if key == "joint" else "normal")
    ax.axhline(1.0, color=GREY, lw=0.9, ls="--")
    ax.set_xticks(range(len(states)), [lab for _, lab in states], fontsize=8.5)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel("damage at the updated state\n÷ the same at the unchanged model")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.set_title("The median single piece does the same damage as before; the whole mask does more",
                 fontsize=10)
    save(fig, "log_stage0.png")


def main() -> None:
    out = load("gamma_refit.json")
    fig_control(out, load("gamma_refit_v2.json"))
    fig_split(out)
    fig_sparsity(out)
    fig_stage0(out)


if __name__ == "__main__":
    main()
