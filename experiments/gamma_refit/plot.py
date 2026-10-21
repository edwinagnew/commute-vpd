"""Figures for gamma_refit from results/gamma_refit.json (or the smoke file with --smoke).

  pgd_vs_step.png     one question: does refitting Γ on the split state bring it back under the Pile adversary? The
                      three Pile ratios to W0 of the least-squares split state at every refit checkpoint, with the W0
                      floor band from the same job, and the same refit on W0 (the control) on the same axes.
  gamma_changes.png   one question: what did Γ change to get there? The sparsity-loss ratio to W0 and the Jaccard of
                      the trigger's switched-on set with the released Γ's, at the same checkpoints.
  refit_loss.png      one question: did the refit converge? Each term of Γ's own training loss as a 50-step running
                      mean relative to its first ten steps.
  protocol_vs_step.png one question: does a warmed adversary and the released end-of-schedule rate keep W₀ at the floor
                      (the v2 protocol, results/gamma_refit_v2.json, stopped at step 300)? W₀'s adversarial ratio under
                      E15's refit and under v2 on the same axes, the split state for reference.
  stage0_ablation.png one question: do the subcomponents the adversary turns down matter more at the updated model
                      than at W0? Single-subcomponent ablation KL at the state against the same at W0, per state and
                      group, with the "unchanged" diagonal.

Usage (repo root):  uv run python experiments/gamma_refit/plot.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
GREY = "#7f7f7f"
# One colour per state, the same in every figure.
STATE = {"ls": ("least-squares split", "#eb6834"), "w0": ("W₀, no update", "#2a78d6"), "nothing": ("do-nothing", "#7f7f7f"),
         "ct": ("component training", "#2a9d8f"), "adjoint": ("adjoint split", "#8e44ad")}
ARM = {"P": "Pile only", "M": "Pile + task windows"}
CHECK = {"pgd_kl": ("adversarial", "-", 2.2), "stochastic_kl": ("stochastic", "--", 1.2), "ci_masked_kl": ("CI-masked", ":", 1.2)}
GROUP = {"down_changed": ("turned down, most changed", "#eb6834"), "up_changed": ("left up, most changed", "#2a78d6"),
         "down_random": ("turned down, random", "#7f7f7f")}


def load(smoke: bool) -> dict[str, Any]:
    return json.loads((HERE / "results" / f"gamma_refit{'_smoke' if smoke else ''}.json").read_text())


def steps_of(refit: dict[str, Any]) -> list[int]:
    return sorted(int(s) for s in refit["checkpoints"])


def xpos(steps: list[int]) -> np.ndarray:
    """Checkpoints on a log axis, with step 0 (the released Γ) one decade to the left of the first checkpoint."""
    first = min(s for s in steps if s > 0)
    return np.array([first / 10 if s == 0 else s for s in steps], dtype=float)


def step_axis(ax: Any, steps: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(xpos(steps), [("0\n(released Γ)" if s == 0 else str(s)) for s in steps])
    ax.minorticks_off()
    ax.set_xlabel("refit step")
    ax.grid(axis="y", lw=0.3, alpha=0.5)


def tag_label(tag: str) -> str:
    name, arm = tag.rsplit("_", 1)
    return f"{STATE[name][0]}, {ARM[arm]}"


def pgd_figure(out: dict[str, Any], suffix: str) -> None:
    refits = out["refits"]
    line = 1 + 3 * out["W0"]["pgd_floor"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for tag, r in refits.items():
        steps = steps_of(r)
        colour = STATE[tag.rsplit("_", 1)[0]][1]
        for key, (word, ls, lw) in CHECK.items():
            ys = [r["checkpoints"][str(s)]["pile"][key] for s in steps]
            ax.plot(xpos(steps), ys, ls, color=colour, lw=lw, marker="o", ms=5 if key == "pgd_kl" else 3,
                    label=f"{tag_label(tag)}: {word}")
    ax.axhspan(1, line, color=GREY, alpha=0.25, lw=0)
    ax.text(xpos([0, 100])[0] * 0.95, line, f"W₀ floor band (to {line:.2f}: 3 seeds of the adversary at W₀)", fontsize=7.5, color=GREY,
            ha="left", va="bottom")
    if "w0_P" not in refits:
        ax.text(0.98, 0.5, "W₀ control (the same refit on the unchanged model): not run", transform=ax.transAxes, fontsize=8,
                ha="right", va="center", color="#b03a2e")
    ax.set_yscale("log")
    ax.set_ylabel("Pile reconstruction KL ÷ the same check at W₀")
    step_axis(ax, steps_of(next(iter(refits.values()))))
    ax.legend(fontsize=7.5, loc="center left")
    ax.set_title("Does refitting Γ bring the split state under the Pile adversary, and does it leave W₀ at the floor?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"pgd_vs_step{suffix}.png", dpi=150)
    print(FIGURES / f"pgd_vs_step{suffix}.png")


def changes_figure(out: dict[str, Any], suffix: str) -> None:
    refits = out["refits"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for tag, r in refits.items():
        steps = steps_of(r)
        colour = STATE[tag.rsplit("_", 1)[0]][1]
        a.plot(xpos(steps), [r["checkpoints"][str(s)]["sparsity_ratio"] for s in steps], "-o", color=colour, label=tag_label(tag))
        sizes = [r["checkpoints"][str(s)]["s_on_size"] for s in steps]
        if tag.startswith("w0"):  # W₀ has no switched-on set under the released Γ, so no Jaccard; report the sizes only
            b.text(0.03, 0.04, f"{tag_label(tag)}: no released set; at most {max(sizes)} subcomponents switched on", transform=b.transAxes,
                   fontsize=7.5, color=colour)
            continue
        jac = [r["checkpoints"][str(s)]["s_on_jaccard_released"] for s in steps]
        b.plot(xpos(steps), jac, "-o", color=colour, label=tag_label(tag))
        for x, y, n in zip(xpos(steps), jac, sizes):
            b.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    a.axhline(1.0, color=GREY, lw=0.8, ls="--")
    a.text(xpos([0, 100])[0], 1.0, "released Γ at W₀ reads 1", fontsize=7.5, color=GREY, va="bottom")
    a.axhline(1.05, color="#b03a2e", lw=0.8, ls=":")
    a.text(xpos([0, 100])[0], 1.05, "filed line: above 1.05, Γ is calling more subcomponents important", fontsize=7.5, color="#b03a2e", va="bottom")
    a.set_ylabel("VPD sparsity loss on Pile ÷ W₀'s under the released Γ")
    a.set_title("How many subcomponents Γ calls important", fontsize=9.5)
    b.set_ylim(0, 1.08)
    b.set_ylabel("Jaccard with the released Γ's switched-on set")
    b.set_title("Does Γ still switch on the same set at the trigger?\n(numbers: size of the switched-on set)", fontsize=9.5)
    for ax in (a, b):
        step_axis(ax, steps_of(next(iter(refits.values()))))
        ax.legend(fontsize=7.5)
    fig.suptitle("What did Γ change while the adversarial ratio fell?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"gamma_changes{suffix}.png", dpi=150)
    print(FIGURES / f"gamma_changes{suffix}.png")


def loss_figure(out: dict[str, Any], suffix: str) -> None:
    terms = {"PersistentPGDReconLoss": ("persistent-adversary reconstruction KL", "#c0392b", "-"),
             "StochasticReconSubsetLoss": ("stochastic-mask reconstruction KL", "#2a78d6", "--"),
             "ImportanceMinimalityLoss": ("importance minimality (sparsity)", "#eb6834", ":")}
    refits = out["refits"]
    fig, axes = plt.subplots(1, len(refits), figsize=(6 * len(refits), 4.2), squeeze=False)
    for ax, (tag, r) in zip(axes[0], refits.items()):
        curve = r["curve"]
        step = np.array([c["step"] for c in curve])
        n = min(50, max(1, len(curve) // 10))
        for key, (word, colour, ls) in terms.items():
            y = np.array([c[key] for c in curve])
            base = float(np.median(y[: min(10, len(y))]))  # the median: step 2's spike would set a mean
            smooth = np.convolve(y / base, np.ones(n) / n, mode="valid")
            ax.plot(step[n - 1 :], smooth, ls, color=colour, lw=1.6, label=word)
        ax.axhline(1.0, color=GREY, lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("refit step")
        ax.set_ylabel(f"loss term ÷ its first-10-step median\n({n}-step running mean)")
        ax.set_title(tag_label(tag), fontsize=9.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.legend(fontsize=7.5)
    fig.suptitle("Did the refit converge? Γ's own training loss, term by term", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"refit_loss{suffix}.png", dpi=150)
    print(FIGURES / f"refit_loss{suffix}.png")


def stage0_figure(out: dict[str, Any], suffix: str) -> None:
    states = [s for s in ("nothing", "ls", "ct") if s in out["stage0"]]
    fig, axes = plt.subplots(1, len(states), figsize=(4.2 * len(states), 4.4), sharex=True, sharey=True)
    lo, hi = 1e-6, 1.0
    for ax, name in zip(np.atleast_1d(axes), states):
        rec = out["stage0"][name]
        for g, (word, colour) in GROUP.items():
            grp = rec["groups"][g]
            x = np.concatenate([np.array(v, dtype=float) for v in grp["kl_w0"].values()])
            y = np.concatenate([np.array(v, dtype=float) for v in grp["kl_state"].values()])
            ax.plot(np.clip(x, lo, hi), np.clip(y, lo, hi), "o", color=colour, ms=5, mec="k", mew=0.3, alpha=0.85,
                    label=f"{word} (median ratio {grp['summary']['median_ratio']:.2f})")
        ax.plot([lo, hi], [lo, hi], "-", color=GREY, lw=0.8)
        ax.plot([lo, hi], [2 * lo, 2 * hi], ":", color="#b03a2e", lw=0.8)
        ax.text(2e-6, 5e-6, "unchanged", fontsize=7, color=GREY, rotation=45)
        ax.text(2e-6, 1.2e-5, "filed line: 2× (matters more at the state)", fontsize=7, color="#b03a2e", rotation=45)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("ablation KL at W₀ (nats)")
        ratio = rec["pgd_kl"] / out["stage0"]["w0"]["pgd_kl"]
        ax.set_title(f"{STATE[name][0]}\n(the adversary does {ratio:.2f}× its W₀ damage here)", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(lw=0.3, alpha=0.5)
    np.atleast_1d(axes)[0].set_ylabel("ablation KL at the state (nats)")
    fig.suptitle("Stage 0: do the subcomponents the adversary turns down matter more at the updated model than at W₀?\n"
                 "48 subcomponents per group; single-subcomponent ablation KL on 8 validation rows", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"stage0_ablation{suffix}.png", dpi=150)
    print(FIGURES / f"stage0_ablation{suffix}.png")


def protocol_figure(out: dict[str, Any], suffix: str) -> None:
    v2_path = HERE / "results" / "gamma_refit_v2.json"
    if suffix or not v2_path.is_file():
        return
    v2 = json.loads(v2_path.read_text())["partial"]["w0_P"]  # stopped at step 300 by the early-stop rule
    line = 1 + 3 * out["W0"]["pgd_floor"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    series = [("W₀ under E15's refit (lr 5e-5, cold adversary)", out["refits"]["w0_P"], STATE["w0"][1], "-"),
              ("W₀ under the v2 protocol (300-step adversary warm-up, lr 5e-6; stopped at step 300)", v2, STATE["w0"][1], "--"),
              ("least-squares split under E15's refit, for reference", out["refits"]["ls_P"], STATE["ls"][1], "-")]
    all_steps = steps_of(out["refits"]["w0_P"])
    for label, r, colour, ls in series:
        steps = steps_of(r)
        ax.plot(xpos(steps), [r["checkpoints"][str(s)]["pile"]["pgd_kl"] for s in steps], ls, color=colour, marker="o", ms=5,
                lw=2.2 if "W₀" in label else 1.2, alpha=1 if "W₀" in label else 0.6, label=label)
    ax.axhspan(1, line, color=GREY, alpha=0.25, lw=0)
    ax.text(xpos([0, 100])[0] * 0.95, line, f"W₀ floor band (to {line:.2f})", fontsize=7.5, color=GREY, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylabel("adversarial Pile KL ÷ the same check at W₀")
    step_axis(ax, all_steps)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Does a warmed adversary and the end-of-schedule rate keep W₀ at the floor?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "protocol_vs_step.png", dpi=150)
    print(FIGURES / "protocol_vs_step.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    suffix = "_smoke" if args.smoke else ""
    out = load(args.smoke)
    FIGURES.mkdir(exist_ok=True)
    pgd_figure(out, suffix)
    changes_figure(out, suffix)
    loss_figure(out, suffix)
    stage0_figure(out, suffix)
    protocol_figure(out, suffix)


if __name__ == "__main__":
    main()
