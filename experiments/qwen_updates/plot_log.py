"""The research log's three figures for E45, drawn from saved results with plain labels (no model, no new run).

  log_anchor2.png  the invented-facts fine-tune (anchor 2) over training, both seeds: first, KL to start at each saved
                   state on its own text and on the maths rows, with gold beside it (E_*.json states), the grid values
                   and the matching band (predictions_amend2.json); then the share of held-out invented facts it
                   recalls (E_a2long_s*.json gen.facts.invented, the original model's from control.json); last, its greedy GSM8K accuracy without a prompt (gen.test.strict_dropped) and with
                   a 4-shot prompt at steps 100 and 300 (check_format.json). The steps where the plan's equal-movement
                   points fell (2 to 12, E_a2long_s*.json chosen) are shaded.
  log_step2.png    KL to the original model on the shared maths rows at step 2, the first step with a nonzero
                   rate, per fine-tune (T_*.json checks), against the old limit (predictions.json early.step2_kl_max),
                   the new one (predictions_amend2.json) and the one-step control it is a tenth of (control.json).
  log_anchor1.png  the change in the original model's GSM8K accuracy on 512 training problems under each candidate
                   prompt for anchor 1 against no prompt, paired on the same problems, two standard errors (P.json).

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/qwen_updates/plot_log.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d4"
SEED = {"0": "#2a78d6", "1": "#eb6834"}
GREY_BAND = "#efeeea"
GOLD = "#3a8f4f"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / f"{name}.json").read_text())


def style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"log_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def grid_steps(e: dict[str, Any]) -> list[int]:
    return sorted({c["step"] for c in e["chosen"]})


def kl_panel(ax: Axes) -> None:
    """KL to start at each saved state, on the fine-tune's own text (solid) and the shared maths rows (dashed)."""
    dp = load("predictions_amend2")["rules"]["decisive_point"]
    target, tol = dp["target_kl_maths"], dp["tol_ratio"]
    ax.axhspan(target / tol, target * tol, color="#dcebdc", zorder=0)
    ax.text(330, target * 1.05, "matching band on the\nmaths rows (0.62 ± ×1.5)", fontsize=7.5, color="#2f6b3a", va="center")
    for g in (0.01, 0.03, 0.1, 0.3):
        ax.axhline(g, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.text(330, 0.03, "equal-movement\npoints 0.01-0.3", fontsize=7.5, color=MUTED, va="center")
    runs = {"invented facts, seed 0": ("E_a2long_s0", SEED["0"]), "invented facts, seed 1": ("E_a2long_s1", SEED["1"]),
            "gold solutions": ("E_gold_s0", GOLD)}
    for lab, (name, col) in runs.items():
        st = load(name)["states"]
        steps = sorted(int(k) for k in st if int(k) >= 2)
        ax.plot(steps, [st[str(t)]["kl"]["own"] for t in steps], "o-", ms=3.5, lw=1.2, color=col, label=lab)
        ax.plot(steps, [st[str(t)]["kl"]["maths"] for t in steps], "s--", ms=3.5, lw=1.2, color=col, mfc="white")
    ax.plot([], [], "k-", lw=1.2, label="solid: on its own training text")
    ax.plot([], [], "k--", lw=1.2, label="dashed: on the shared maths rows")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 10)
    ax.set_ylabel("KL to the original model")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=2)


def fig_anchor2() -> None:
    ctrl, fmt = load("control"), load("check_format")["states"]
    runs = {s: load(f"E_a2long_s{s}") for s in SEED}
    steps_grid = sorted({t for e in runs.values() for t in grid_steps(e)})
    fig, (kl, top, bot) = plt.subplots(3, 1, figsize=(8.4, 9.4), sharex=True,
                                       gridspec_kw={"height_ratios": [1.15, 1, 1]})
    for ax in (kl, top, bot):
        ax.axvspan(steps_grid[0] * 0.9, steps_grid[-1] * 1.1, color=GREY_BAND, zorder=0)
        ax.set_xscale("log")
        style(ax)
    kl_panel(kl)
    kl.set_title("The invented-facts fine-tune reaches every equal-movement point by step 12 but learns its facts only by\n"
                 "step 100; its unprompted maths accuracy falls there, and mostly returns under a 4-shot prompt",
                 fontsize=10)
    top.text(4.6, 0.5, "steps where the invented-facts\nrun's equal-movement points fell", ha="center", fontsize=8.5,
             color=MUTED)
    top.text(45, 0.12, "no state saved\nbetween steps 12 and 100", ha="center", fontsize=8, color=MUTED)
    for s, e in runs.items():
        steps = sorted(int(k) for k in e["gen"])
        top.plot(steps, [e["gen"][str(t)]["facts"]["invented"] for t in steps], "o--", lw=1.3, color=SEED[s],
                 label=f"seed {s}")
        bot.plot(steps, [e["gen"][str(t)]["test"]["strict_dropped"] for t in steps], "o--", lw=1.3, color=SEED[s],
                 label=f"seed {s}, no prompt")
        for t in (100, 300):
            v = fmt[f"a2long_s{s}_step_{t}"]["strict"]
            bot.plot(t, v, "D", mfc="white", mew=1.6, ms=8, color=SEED[s],
                     label=f"seed {s}, 4-shot prompt" if t == 100 else None)
            bot.text(t * (0.8 if s == "0" else 1.07), v, f"{v:.2f}", fontsize=7.5, color=SEED[s], va="center",
                     ha="right" if s == "0" else "left")
    w0_facts = max(ctrl["facts_s0"]["invented"], ctrl["facts_s1"]["invented"])
    top.axhline(w0_facts, color=MUTED, ls="--", lw=1)
    top.text(130, w0_facts + 0.03, "original model (chance)", fontsize=8, color=MUTED)
    top.set_ylim(-0.02, 1.05)
    top.set_ylabel("held-out invented facts recalled")
    top.legend(frameon=False, fontsize=8.5, loc="upper left")
    w0_plain, w0_few = ctrl["W0"]["first256_strict_dropped"], fmt["W0"]["strict"]
    bot.axhline(w0_plain, color=MUTED, ls="--", lw=1)
    bot.text(2.05, w0_plain + 0.05, f"original model, no prompt ({w0_plain:.2f})", fontsize=8, color=MUTED)
    bot.axhline(w0_few, color=MUTED, ls=":", lw=1)
    bot.text(2.05, w0_few - 0.06, f"original model, 4-shot prompt ({w0_few:.2f})", fontsize=8, color=MUTED)
    bot.set_ylim(0, 1)
    bot.set_ylabel("GSM8K accuracy (greedy,\n256 test problems)")
    bot.set_xlabel("training step (log scale)")
    bot.set_xticks([2, 3, 4, 6, 12, 100, 300], ["2", "3", "4", "6", "12", "100", "300"])
    bot.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, "anchor2")


def fig_step2() -> None:
    arms = (("worked solutions\n(gold)", "T_gold_s0"), ("invented facts\nseed 0", "T_a2long_s0"),
            ("invented facts\nseed 1", "T_a2long_s1"), ("its own correct\nanswers", "T_own_s0"),
            ("self-distillation null\n(own answers, unfiltered)", "T_null_s0"))
    old = load("predictions")["early"]["step2_kl_max"]
    new = load("predictions_amend2")["early"]["step2_kl_max"]
    one_step = load("control")["one_step"]["kl_to_start"]
    vals = [load(f)["checks"]["step 2 KL on the maths rows"]["kl"] for _, f in arms]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, v in enumerate(vals):
        c = "#2a78d6" if v <= old else "#c0392b"
        ax.plot([i, i], [6e-4, v], color=c, lw=2)
        ax.plot(i, v, "o", color=c, ms=9)
        ax.text(i + 0.12, v, f"{v:.4f}", va="center", fontsize=8.5)
    for y, lab, ls in ((old, f"old limit {old:.3f}, set from the gold run: red stopped", "--"),
                       (new, f"new limit {new:.4f}, a tenth of the line above", "-"),
                       (one_step, f"one full-rate step with no warm-up: {one_step:.3f}", ":")):
        ax.axhline(y, color=INK if ls == "-" else MUTED, ls=ls, lw=1.1)
        ax.text(-0.45, y * 1.1, lab, ha="left", fontsize=8.5, color=INK if ls == "-" else MUTED)
    ax.set_yscale("log")
    ax.set_ylim(6e-4, 0.2)
    ax.set_xlim(-0.5, len(arms) - 0.4)
    ax.set_xticks(range(len(arms)), [a for a, _ in arms], fontsize=8.5)
    ax.set_ylabel("KL to the original model on the shared maths\nrows at step 2 (first step, a tenth of full rate)")
    ax.set_title("Two fine-tunes stopped at a first-step limit set from the gold run alone;\n"
                 "the new limit, set after the stop, is tied to the no-warm-up failure instead", fontsize=10)
    style(ax)
    fig.tight_layout()
    save(fig, "step2")


def fig_anchor1() -> None:
    """Paired change in accuracy against no prompt on the same 512 problems, with two paired standard errors."""
    runs = load("P")["runs"]
    names = (("fewshot", "4 worked examples\nfirst (few-shot)"), ("recheck", "re-check every\narithmetic step"),
             ("givens", "list the givens\nfirst"), ("format", "answer-template\nprompt (fallback)"))
    base = [float(b) for b in runs["none"]["strict"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (k, _) in enumerate(names):
        d = [float(a) - b for a, b in zip(runs[k]["strict"], base, strict=True)]
        m = sum(d) / len(d)
        se = (sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) ** 0.5
        ax.bar(i, m, 0.6, color="#eb6834" if k == "format" else "#2a78d6")
        ax.errorbar(i, m, yerr=2 * se, color=INK, capsize=4, lw=1)
        ax.text(i + 0.33, m, f"{m:+.3f}", va="center", fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.9)
    ax.axhline(0.03, color=MUTED, ls="--", lw=1)
    ax.text(3.45, 0.036, "gain a prompt needed (0.03)", ha="right", fontsize=8, color=MUTED)
    fol = load("P")["format"]  # strict rule; the 93% (474/512, whitespace forgiven) is from Amendment 2's jq count
    ax.text(2.55, -0.17, f"template followed: 93% forgiving whitespace,\n{fol['with']:.0%} by the strict rule, "
            f"{fol['without']:.0%} without the prompt", ha="right", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(names)), [lab for _, lab in names], fontsize=8.5)
    ax.set_ylim(-0.2, 0.08)
    ax.set_ylabel(f"change in the original model's accuracy\nagainst no prompt ({sum(base) / len(base):.3f}), "
                  "512 problems")
    ax.set_title("No prompt raises the original model's maths accuracy beyond noise;\n"
                 "the answer-template prompt, the fallback, lowers it", fontsize=10)
    style(ax)
    fig.tight_layout()
    save(fig, "anchor1")


def main() -> None:
    fig_anchor2()
    fig_step2()
    fig_anchor1()


if __name__ == "__main__":
    main()
