"""The research log's four figures for E4, drawn from saved results with plain labels (no model, no new runs).

  log_learn.png        damage elsewhere (held-out KL off the trigger, step 30) of component training with all, 100,
                       10, 1 or the paper's six emoticon pieces free, against the 0.110 limit; reads concentration.json.
  log_switches.png     probability of the target at the trigger in the CI-masked stand-in, with Γ reading the updated
                       model and with Γ's importance values held at the original model's; reads amend1.json.
  log_switched_on.png  share of the switching effect carried by the switched-on set and by a random set of its size;
                       reads amend2.json.
  log_sgd_rate.png     plain SGD's damage elsewhere against its learning rate, filled where it learned; the follow-up
                       grid (experiments/sgd_grid/results/sgd_grid.json) beside E4's (concentration.json), Adam for
                       reference.

Usage (repo root):  python experiments/concentration_test/plot_log.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
SGD_GRID = HERE.parent / "sgd_grid" / "results" / "sgd_grid.json"
KL_OTHER_MAX, P_MIN = 0.110, 0.9  # E4's rate rule (run.py)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, LIGHT, ORANGE = "#2a78d6", "#9cc3ee", "#eb6834"
ARM = {
    "do_nothing": "do-nothing\n(pieces unchanged)",
    "adam_all": "all 9,881\npieces free",
    "top100": "100 pieces\nfree",
    "top10": "10 pieces\nfree",
    "top1": "1 piece\nfree",
    "siblings": "the paper's 6\nemoticon pieces",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def style(ax: Any, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)


def end(arm: dict[str, Any]) -> dict[str, float]:
    return arm["runs"][arm["selected_lr"]]["curve"][-1]


def fig_learn(arms: dict[str, Any]) -> None:
    keys = ("adam_all", "top100", "top10", "top1", "siblings")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, k in enumerate(keys):
        e = end(arms[k])
        ax.barh(i, e["heldout_kl_other"], 0.62, color=MUTED if k == "adam_all" else BLUE)
        label = f"{e['heldout_kl_other']:.3f}   (held-out target probability {e['heldout_p_o']:.2f})"
        ax.text(e["heldout_kl_other"] + 0.002, i, label, va="center", fontsize=8)
    ax.axvline(KL_OTHER_MAX, color=INK, ls=":", lw=1)
    ax.text(KL_OTHER_MAX, -0.75, "damage limit ", ha="right", va="center", fontsize=8)
    ax.set_yticks(range(len(keys)), [ARM[k].replace("\n", " ") for k in keys], fontsize=8.5)
    ax.set_ylim(len(keys) - 0.5, -1.0)
    ax.set_xlim(0, 0.12)
    ax.set_xlabel("damage elsewhere: held-out KL away from the trigger, after 30 steps")
    ax.set_title("A few pieces learn the target with damage well under the limit", fontsize=10)
    style(ax, "x")
    save(fig, "log_learn.png")


def fig_switches(a1: dict[str, Any]) -> None:
    keys = ("do_nothing", "adam_all", "top100", "top10", "top1", "siblings")
    rows = {"do_nothing": {"state_gated": a1["do_nothing_state_gated"], "w0_gated": a1["do_nothing_w0_gated"]},
            **{k: a1["arms"][k] for k in keys[1:]}}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.38
    for i, k in enumerate(keys):
        for j, (g, lab, col) in enumerate((("state_gated", "Γ reads the updated model (switches can change)", BLUE),
                                            ("w0_gated", "Γ's switches held at the original model's", LIGHT))):
            v = rows[k][g]["p_o_ci"]
            ax.bar(i + (j - 0.5) * w, v, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(i + (j - 0.5) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(range(len(keys)), [ARM[k] for k in keys], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("probability of the target at the trigger,\npieces only (masked by Γ, Δ left out)")
    ax.set_title("With Γ's switches held at the original's, the trained pieces reach at most 0.37, not 0.98", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    style(ax)
    save(fig, "log_switches.png")


def fig_switched_on(a2: dict[str, Any]) -> None:
    keys = ("do_nothing", "top1", "top10", "siblings")
    labels = {**ARM, "do_nothing": "do-nothing and\nall pieces free\n(same switches)"}
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    w = 0.38
    for i, k in enumerate(keys):
        a = a2["arms"][k]
        for j, (v, lab, col) in enumerate(((a["recovery"]["0.3"], "the switched-on set", BLUE),
                                           (a["recovery_random_control"], "a random set of the same size", MUTED))):
            ax.bar(i + (j - 0.5) * w, v, w * 0.92, color=col, label=lab if i == 0 else None)
            ax.text(i + (j - 0.5) * w, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
        ax.text(i - 0.5 * w, 0.03, f"{a['n_on']['0.3']}\npieces", ha="center", fontsize=7.5, color="white")
    ax.set_xticks(range(len(keys)), [labels[k] for k in keys], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the switching effect carried\n(0 none, 1 all)")
    ax.set_title("The switched-on set carries 0.45 to 0.64 of the switching effect; a random set carries none", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)
    save(fig, "log_switched_on.png")


def fig_sgd_rate(grid: dict[str, Any], arms: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for runs, lab, marker in ((grid["runs"], "follow-up grid (2026-10-21)", "o"),
                              (arms["sgd_all"]["runs"], "E4's grid", "s")):
        lrs = sorted(runs, key=float)
        ends = [runs[lr]["curve"][-1] for lr in lrs]
        x, y = [float(lr) for lr in lrs], [e["heldout_kl_other"] for e in ends]
        ax.plot(x, y, color=BLUE, lw=0.8, ls="-" if marker == "o" else "--")
        learned = [e["heldout_p_o"] >= P_MIN for e in ends]
        ax.scatter([a for a, ok in zip(x, learned, strict=True) if ok], [b for b, ok in zip(y, learned, strict=True) if ok],
                   marker=marker, s=40, color=BLUE, zorder=3, label=f"{lab}: target learned")
        ax.scatter([a for a, ok in zip(x, learned, strict=True) if not ok],
                   [b for b, ok in zip(y, learned, strict=True) if not ok],
                   marker=marker, s=40, facecolors="white", edgecolors=BLUE, zorder=3, label=f"{lab}: target not learned")
    ax.axhline(KL_OTHER_MAX, color=INK, ls=":", lw=1, label="damage limit")
    ax.axhline(end(arms["adam_all"])["heldout_kl_other"], color=ORANGE, lw=1, label="Adam, all pieces free")
    ax.set(xscale="log", yscale="log", xlabel="plain SGD learning rate",
           ylabel="damage elsewhere: held-out KL away\nfrom the trigger, after 30 steps")
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10], ["0.001", "0.01", "0.1", "1", "10"])
    ax.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax.set_title("Plain SGD learns within the limit at rates 0.003 to 0.1; E4's grid started above them", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(ax)
    save(fig, "log_sgd_rate.png")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    arms = load(RES / "concentration.json")["arms"]
    fig_learn(arms)
    fig_switches(load(RES / "amend1.json"))
    fig_switched_on(load(RES / "amend2.json"))
    fig_sgd_rate(load(SGD_GRID), arms)


if __name__ == "__main__":
    main()
