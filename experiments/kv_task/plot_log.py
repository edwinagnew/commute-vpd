"""The research log's four figures for E10, redrawn from saved results with plain labels (no model, no new run).

  log_learning.png  held-out probability of the right next word after marker and key, along component training, for
                    the three tasks (results/kv_task.json, each task's pullback curve).
  log_check.png     the adversarial ratio on Pile text at each task's first learned step, component training beside
                    the least-squares split of the weight fine-tune (results/kv_task.json), with the pass and fail
                    lines from the unchanged decomposition's spread under three adversary seeds
                    (results/kv_task_amend1.json, W0 floor).
  log_routes.png    where component training's gain on the target comes from: pieces Γ switches on, pieces Γ keeps
                    off, the rest; with the weight fine-tune's switched-on share (do-nothing = split) as a tick
                    (results/kv_task.json, routes).
  log_damage.png    the 32-pair cipher and the copy control at steps 10, 30, 100 and 300: adversarial ratio against
                    damage elsewhere and against size of change (results/kv_task_amend1.json; the copy control's
                    step 30 from results/kv_task.json); the earlier 0.2-nat damage cut as a vertical line.

Usage (repo root):  python experiments/kv_task/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from plot import FIGURES, RESULTS, arm_eval

INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d4"
BLUE, RED, ORANGE, GOLD = "#2a78d6", "#c8463d", "#eb6834", "#c9a13b"
TASKS = ("rand1", "rand32", "ident32")
NAME = {"rand1": "cipher, 1 pair", "rand32": "cipher, 32 pairs", "ident32": "copy control, 32 words"}
COLOUR = {"rand1": GOLD, "rand32": BLUE, "ident32": ORANGE}
# step-label offsets (points) where the default, up and right, would collide
LABEL_AT = {("damage", "rand32", 10): (6, -12), ("damage", "rand32", 300): (6, -12),
            ("damage", "ident32", 300): (-12, 7), ("damage", "ident32", 100): (-6, 7),
            ("size", "ident32", 100): (-20, 7), ("size", "ident32", 300): (6, -12),
            ("size", "rand32", 300): (-14, -13)}


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    out = FIGURES / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out)


def lines(amend: dict[str, Any]) -> tuple[float, float, tuple[float, float]]:
    """Pass and fail lines and the unchanged decomposition's band under three adversary seeds."""
    w0 = amend["shards"][0]["W0"]
    f = w0["pgd_floor"]
    seeds = [w0["pile"][k]["pgd_kl"] / w0["pile"]["0"]["pgd_kl"] for k in ("0", "1", "2")]
    return 1.5 / (1 + f), 1.5 * (1 + f), (min(seeds), max(seeds))


def fig_learning(tasks: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for t in TASKS:
        pb = tasks[t]["pullback"]
        ax.plot([c["step"] for c in pb["curve"]], [c["heldout_p_o"] for c in pb["curve"]], color=COLOUR[t], lw=2,
                label=f"{NAME[t]} (first read at step {pb['scored_step']})")
        if t != "rand1":  # the saved states the follow-up re-read
            pts = [c for c in pb["curve"] if c["step"] in (10, 30, 100, 300)]
            ax.scatter([c["step"] for c in pts], [c["heldout_p_o"] for c in pts], s=30, color=COLOUR[t],
                       edgecolor="white", lw=1, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.text(1.05, 0.52, "learned (0.5)", color=MUTED, fontsize=8)
    ax.text(10.5, 0.36, "copy at step 10: 0.39", color=COLOUR["ident32"], fontsize=7.5, ha="left")
    ax.set_xscale("log")
    ax.set_xlim(1, 300)
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("probability of the right word\non held-out text")
    ax.set_title("Component training learns all three tasks within 30 steps", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "log_learning.png")


def fig_check(tasks: dict[str, Any], amend: dict[str, Any]) -> None:
    pass_line, fail_line, _ = lines(amend)
    arms = (("pullback", "component training", BLUE), ("lift", "least-squares split of the weight fine-tune", RED))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.36
    for j, (arm, lab, col) in enumerate(arms):
        for i, t in enumerate(TASKS):
            v = arm_eval(tasks[t], arm)["pile"]["ratios"]["pgd_kl"]
            x = i + (j - 0.5) * w
            ax.bar(x, v - 1, w * 0.92, bottom=1, color=col, label=lab if i == 0 else None, zorder=2)
            ax.text(x, v * 1.06, f"{v:.2f}" if v < 5 else f"{v:.1f}", ha="center", fontsize=8.5)
    ax.axhspan(pass_line, fail_line, color=GRID, alpha=0.6, lw=0, zorder=0, label="inconclusive")
    ax.axhline(pass_line, color=MUTED, lw=0.9, ls="--", zorder=1, label=f"pass line {pass_line:.2f}")
    ax.axhline(fail_line, color=MUTED, lw=0.9, ls=":", zorder=1, label=f"fail line {fail_line:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 70)
    ax.set_xticks(range(len(TASKS)), [NAME[t] for t in TASKS], fontsize=9)
    ax.set_ylabel("adversarial ratio on Pile text\n(1 = as faithful as before; log scale)")
    ax.set_title("At the first learned step, component training passes on one pair and on copying, fails on 32 pairs",
                 fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "log_check.png")


def fig_routes(tasks: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    parts = (("r", "pieces Γ switches on (re-switching share)", "#5b8fc9"),
             ("e_gate", "pieces Γ keeps switched off (masked-off share)", "#e3a25c"))
    for i, t in enumerate(TASKS):
        sh = tasks[t]["pullback"]["eval"]["routes"]["shares"]
        left = 0.0
        for key, lab, col in parts:
            ax.barh(i, sh[key], left=left, color=col, height=0.6, label=lab if i == 0 else None)
            ax.text(left + sh[key] / 2, i, f"{sh[key]:.2f}", ha="center", va="center", fontsize=8.5)
            left += sh[key]
        ax.barh(i, 1 - left, left=left, color=GRID, height=0.6, label="the rest (the pieces' own change, and Δ)" if i == 0 else None)
        r_ft = tasks[t]["do_nothing"]["routes"]["shares"]["r"]  # the same under the split: Γ sees the same weights
        ax.plot([r_ft, r_ft], [i - 0.38, i + 0.38], color=INK, lw=2, zorder=3,
                label="re-switching share of the weight fine-tune\n(under do-nothing and the split alike)" if i == 0 else None)
    ax.set_yticks(range(len(TASKS)), [NAME[t] for t in TASKS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of component training's gain on the right word")
    ax.set_title("Copying runs through pieces Γ switches on; the cipher through pieces Γ keeps off", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "log_routes.png")


def damage_states(tasks: dict[str, Any], amend: dict[str, Any]) -> list[dict[str, Any]]:
    states = [{"task": s["task"], "step": s["step"], "size": s["size"], "damage": s["heldout"]["kl_other"],
               "ratio": s["pile"]["pgd_kl"]} for sh in amend["shards"] for s in sh["states"].values()]
    pb = tasks["ident32"]["pullback"]
    at30 = next(c for c in pb["curve"] if c["step"] == 30)
    states.append({"task": "ident32", "step": 30, "size": at30["rel_norm_max"], "damage": at30["heldout_kl_other"],
                   "ratio": pb["eval"]["pile"]["ratios"]["pgd_kl"]})
    return states


def fig_damage(tasks: dict[str, Any], amend: dict[str, Any]) -> None:
    pass_line, fail_line, band = lines(amend)
    states = damage_states(tasks, amend)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    xs = (("damage", "damage elsewhere: change in the model's other\npredictions, in nats (log scale)"),
          ("size", "size of change: largest relative change\nof any weight matrix"))
    for ax, (key, xlabel) in zip(axes, xs, strict=True):
        ax.axhspan(*band, color=GRID, lw=0, label="unchanged decomposition, three adversary seeds")
        ax.axhline(pass_line, color=MUTED, lw=0.9, ls="--", label=f"pass line {pass_line:.2f}")
        ax.axhline(fail_line, color=MUTED, lw=0.9, ls=":", label=f"fail line {fail_line:.2f}")
        for t in ("rand32", "ident32"):
            pts = sorted((s for s in states if s["task"] == t), key=lambda s: s["step"])
            ax.scatter([s[key] for s in pts], [s["ratio"] for s in pts], s=60, color=COLOUR[t], edgecolor="white",
                       lw=1.2, zorder=3, label=NAME[t])
            for s in pts:
                ax.annotate(f"step {s['step']}", (s[key], s["ratio"]), xytext=LABEL_AT.get((key, t, s["step"]), (6, 5)),
                            textcoords="offset points", fontsize=7.5, color=COLOUR[t])
        ax.set_xlabel(xlabel, fontsize=9)
        style(ax)
    axes[0].axvline(0.2, color=INK, lw=0.8, ls="-.", label="earlier damage cut, 0.2 nats")
    axes[0].set_xscale("log")
    axes[0].set_ylabel("adversarial ratio on Pile text\n(1 = as faithful as before)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("The one failing state is the 32-pair cipher at step 30, where its damage elsewhere peaks;\n"
                 "later states of the same run are larger in size and pass", fontsize=10.5)
    fig.tight_layout()
    save(fig, "log_damage.png")


def main() -> None:
    tasks = load("kv_task.json")["tasks"]
    amend = load("kv_task_amend1.json")
    FIGURES.mkdir(exist_ok=True)
    fig_learning(tasks)
    fig_check(tasks, amend)
    fig_routes(tasks)
    fig_damage(tasks, amend)


if __name__ == "__main__":
    main()
