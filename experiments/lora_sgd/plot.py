"""Figures for lora_sgd (E27), read from results/lora_sgd.json and E14's experiments/lora_split/results/lora_split.json:
  outside.png   outside-both share / its random-update null along the nine checkpoints: the SGD LoRA, the in-job Adam
                LoRA, E14's Adam LoRA and full fine-tune and E22's SGD full fine-tune, with the random reference (1) and P1's line (0.7)
  states.png    adversarial ratio of the do-nothing, least-squares and adjoint states at steps 10/100/1000: SGD LoRA, the
                in-job Adam LoRA (10, 1000) and E14's Adam LoRA, with W0 and the same-job verdict band
  grid.png      the SGD rate grid at step 1000: held-out P(target) and KL elsewhere against the rate, with E4's limits and
                the Adam LoRA's reading
  cos.png       mean |cos| of the read factor A with its initialisation along steps, SGD and Adam

Usage (repo root): uv run python experiments/lora_sgd/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
E14 = HERE.parent / "lora_split" / "results" / "lora_split.json"
E22 = HERE.parent / "small_finetune_splits" / "results" / "small_finetune_splits.json"
LINE = 1.5
COLOR = {"sgd": "#2a78d6", "adam": "#eb6834", "ft": "#1baf7a", "ft_sgd": "#7a4fd0"}
OPS = {"nothing": "do-nothing", "least_squares": "least-squares split", "adjoint": "adjoint split"}
INK, MUTED = "#0b0b0b", "#8a8984"


def style(ax: Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def e14_out_out(e14: dict, update: str) -> tuple[list[int], list[float]]:
    T = e14["table"][update]
    steps = sorted(T, key=int)
    return [int(s) for s in steps], [T[s]["pooled_ns"]["out_out"] / T[s]["pooled_ns"]["null_out_out"] for s in steps]


def fig_outside(out: dict, e14: dict, sel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(1, color=MUTED, ls=":", lw=1.2, label="random update of the same size (1)")
    ax.axhline(0.7, color=INK, ls="--", lw=0.8, label="P1's line (0.7)")
    for name, label in (("sgd", f"LoRA, plain SGD {sel} (this run)"), ("adam", "LoRA, Adam 1e-3 (this run)")):
        ser = out["series"].get(name, {})
        x = sorted(map(int, ser))
        ax.plot(x, [ser[str(s)]["blocks"]["pooled_ns"]["out_out_ratio"] for s in x], marker="o", color=COLOR[name], lw=2, label=label)
    for upd, label, color in (("lora24", "LoRA, Adam (E14)", COLOR["adam"]), ("ft", "full fine-tune, Adam (E14)", COLOR["ft"])):
        x, y = e14_out_out(e14, upd)
        ax.plot(x, y, ls="--", marker=".", color=color, lw=1, alpha=0.8, label=label)
    if E22.is_file():
        ser = json.loads(E22.read_text())["series"]["sgd"]
        x = sorted(map(int, ser))
        ax.plot(x, [ser[str(s)]["blocks"]["pooled_ns"]["out_out_ratio"] for s in x], ls="--", marker=".", color=COLOR["ft_sgd"],
                lw=1, alpha=0.8, label="full fine-tune, plain SGD 0.01 (E22)")
    ax.set(xscale="log", xlabel="training step", ylabel="outside-both share / random-update share", ylim=(0, None),
           title="Is the SGD LoRA's update outside the subcomponents as often as noise?")
    style(ax)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "outside.png", dpi=130)


def fig_states(out: dict, e14: dict, sel: str) -> None:
    f = out["W0"]["pgd_floor"]
    e14_ops = {"nothing": "nothing", "least_squares": "lift", "adjoint": "adjoint"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (op, title) in zip(axes, OPS.items(), strict=True):
        ax.axhspan(LINE / (1 + f), LINE * (1 + f), color="#f0efec", label="inconclusive band (same-job W₀ spread)")
        ax.axhline(1, color=MUTED, ls=":", lw=1.2, label="W₀ (unchanged model)")
        ax.axhline(LINE, color=INK, ls="--", lw=1, label="pass line 1.5")
        for name, label in (("sgd", f"LoRA, SGD {sel} (this run)"), ("adam", "LoRA, Adam (this run)")):
            pts = sorted((r["step"], r["pgd"]) for r in out["rows"] if r["lora"] == name and r["op"] == op)
            if pts:
                ax.plot(*zip(*pts, strict=True), marker="o", color=COLOR[name], lw=2, ms=7, label=label)
        P = e14["part2"]["lora24"]
        x = sorted(map(int, P))
        ax.plot(x, [P[str(s)][e14_ops[op]]["pile"]["pgd_kl"] for s in x], ls="--", marker=".", color=COLOR["adam"],
                alpha=0.7, label="LoRA, Adam (E14, other card)")
        ax.set(xscale="log", xticks=[10, 100, 1000], xticklabels=["10", "100", "1000"], title=title, xlabel="training step")
        style(ax)
    axes[0].set_ylabel("adversarial KL ratio to W₀ (Pile, 20-step attack)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Does each placement of the SGD LoRA stay as faithful as the unchanged model?")
    fig.tight_layout()
    fig.savefig(FIGURES / "states.png", dpi=130)


def fig_grid(out: dict) -> None:
    g, last = out["grid"], "1000"
    lrs = sorted(g, key=float)
    x = [float(lr) for lr in lrs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    a.plot(x, [g[lr][last]["p_o"] for lr in lrs], marker="o", color=COLOR["sgd"], lw=2, label="plain SGD LoRA")
    a.axhline(out["rate"]["p_min"], color=INK, ls="--", lw=1, label="E4's rule: P ≥ 0.9")
    a.axhline(out["evals"]["adam"][last]["p_o"], color=COLOR["adam"], ls=":", lw=1.5, label="Adam LoRA (this run)")
    b.plot(x, [g[lr][last]["kl_other"] for lr in lrs], marker="o", color=COLOR["sgd"], lw=2, label="plain SGD LoRA")
    b.axhline(out["rate"]["kl_max"], color=INK, ls="--", lw=1, label="limit: 2 × Adam LoRA (E9)")
    b.axhline(out["evals"]["adam"][last]["kl_other"], color=COLOR["adam"], ls=":", lw=1.5, label="Adam LoRA (this run)")
    sel = float(out["rate"]["selected"])
    for ax in (a, b):
        ax.axvline(sel, color=MUTED, lw=6, alpha=0.25, label=f"selected ({out['rate']['selected']}, {out['rate']['how']})")
        ax.set(xscale="log", xlabel="SGD learning rate")
        style(ax)
    a.set(ylabel="held-out P(target) at step 1000", ylim=(0, 1.02))
    b.set(ylabel="KL elsewhere at step 1000", yscale="log")
    a.legend(fontsize=7.5)
    b.legend(fontsize=7.5)
    fig.suptitle("Which SGD rate learns the task with the least damage elsewhere?")
    fig.tight_layout()
    fig.savefig(FIGURES / "grid.png", dpi=130)


def fig_cos(out: dict, sel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label in (("sgd", f"plain SGD {sel}"), ("adam", "Adam 1e-3")):
        ser = out["series"].get(name, {})
        x = sorted(map(int, ser))
        ax.plot(x, [ser[str(s)]["cos_init"]["mean"] for s in x], marker="o", color=COLOR[name], lw=2, label=f"LoRA, {label}")
    ax.axhline(0.97, color=INK, ls="--", lw=0.8, label="P2's line (0.97)")
    ax.set(xscale="log", xlabel="training step", ylabel="mean |cos| of A's rows with their start",
           title="Does the read factor A stay where it started?")
    style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "cos.png", dpi=130)


def main() -> None:
    out = json.loads((HERE / "results" / "lora_sgd.json").read_text())
    e14 = json.loads(E14.read_text())
    FIGURES.mkdir(exist_ok=True)
    sel = out.get("rate", {}).get("selected", "?")
    fig_outside(out, e14, sel)
    fig_states(out, e14, sel)
    if "rate" in out:
        fig_grid(out)
    fig_cos(out, sel)
    print(f"wrote {sorted(p.name for p in FIGURES.glob('*.png'))}")


if __name__ == "__main__":
    main()
