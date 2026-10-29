"""The research log's four figures for E27, redrawn from saved results with plain labels (no model, no run).

  log_outside.png    outside-both ratio along the nine checkpoints: the SGD LoRA and the Adam LoRA of this run, with the
                     Adam full fine-tune (E14, experiments/lora_split/results/lora_split.json) and the SGD full fine-tune
                     (E22, experiments/small_finetune_splits/results/small_finetune_splits.json); a random update reads 1.
  log_readfactor.png mean |cos| of the read factor A's rows with their random start along the checkpoints, SGD and Adam
                     (results/lora_sgd.json → series.<lora>.<step>.cos_init.mean).
  log_faithful.png   adversarial ratio to the unchanged model of each placement (do-nothing, least-squares split,
                     adjoint split) at steps 10, 100, 1000, drawn from 1; pass and fail lines from this job's spread
                     (results/lora_sgd.json → rows, W0.pgd_floor).
  log_behaviour.png  held-out target probability of each SGD-LoRA placement at steps 10, 100, 1000
                     (results/lora_sgd.json → rows.p_target).

Usage (repo root):  python experiments/lora_sgd/plot_log.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from plot import COLOR, E14, E22, FIGURES, HERE, INK, LINE, MUTED, e14_out_out, style

OPS = (("nothing", "do-nothing"), ("least_squares", "least-squares split"), ("adjoint", "adjoint split"))
STEPS = (10, 100, 1000)
SHADE = {10: 0.35, 100: 0.65, 1000: 1.0}


def load(path: Any) -> dict[str, Any]:
    return json.loads(path.read_text())


def series(ser: dict[str, Any], get: Any) -> tuple[list[int], list[float]]:
    x = sorted(map(int, ser))
    return x, [get(ser[str(s)]) for s in x]


def save(fig: Any, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_outside(out: dict[str, Any], e14: dict[str, Any], e22: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ratio = lambda r: r["blocks"]["pooled_ns"]["out_out_ratio"]
    ax.axhline(1, color=MUTED, ls=":", lw=1.4)
    ax.text(1, 1.03, "random update of the same size", color=MUTED, fontsize=8)
    ax.plot(*series(out["series"]["sgd"], ratio), marker="o", color=COLOR["sgd"], lw=2.2, label="LoRA, plain SGD")
    ax.plot(*series(out["series"]["adam"], ratio), marker="o", color=COLOR["adam"], lw=2.2, label="LoRA, Adam")
    ax.plot(*e14_out_out(e14, "ft"), ls="--", marker=".", color=COLOR["ft"], lw=1.2, label="full fine-tune, Adam (E14)")
    ax.plot(*series(e22["series"]["sgd"], ratio), ls="--", marker=".", color=COLOR["ft_sgd"], lw=1.2,
            label="full fine-tune, plain SGD (E22)")
    ax.set(xscale="log", xlabel="training step", ylim=(0, 1.15),
           ylabel="share of the update the pieces cannot produce,\nover a random update's (1 = noise, 0 = all reachable)")
    ax.set_title("The SGD LoRA sits between noise and the SGD full fine-tune, below the Adam LoRA at most steps",
                 fontsize=9.5)
    style(ax)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, "log_outside.png")


def fig_readfactor(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cos = lambda r: r["cos_init"]["mean"]
    ax.plot(*series(out["series"]["sgd"], cos), marker="o", color=COLOR["sgd"], lw=2.2, label="LoRA, plain SGD")
    ax.plot(*series(out["series"]["adam"], cos), marker="o", color=COLOR["adam"], lw=2.2, label="LoRA, Adam")
    ax.set(xscale="log", xlabel="training step", ylim=(0.8, 1.01),
           ylabel="mean |cos| of the read factor's rows\nwith their random start (1 = unmoved)")
    ax.set_title("Under SGD the read factor stays at its random start; under Adam it moves", fontsize=9.5)
    style(ax)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    save(fig, "log_readfactor.png")


def groups(out: dict[str, Any]) -> list[tuple[str, str, dict[int, dict[str, Any]]]]:
    """One group per (LoRA, placement) that ran: its label, colour and rows by step."""
    res = []
    for lora, name in (("sgd", "SGD LoRA"), ("adam", "Adam LoRA")):
        for op, label in OPS:
            rows = {r["step"]: r for r in out["rows"] if r["lora"] == lora and r["op"] == op}
            if rows:
                res.append((f"{name}\n{label}", COLOR[lora], rows))
    return res


def fig_faithful(out: dict[str, Any]) -> None:
    f = out["W0"]["pgd_floor"]
    lo, hi = LINE / (1 + f), LINE * (1 + f)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.axhspan(lo, hi, color="#f0efec")
    ax.axhline(lo, color=INK, ls="--", lw=1)
    ax.text(4.6, lo - 0.015, f"pass at or below {lo:.2f}", ha="right", va="top", fontsize=8)
    ax.text(4.6, (lo + hi) / 2, "inconclusive", ha="right", va="center", fontsize=8, color=MUTED)
    ax.text(4.6, hi + 0.01, f"fail at or above {hi:.2f}", ha="right", va="bottom", fontsize=8)
    w = 0.26
    for i, (label, color, rows) in enumerate(groups(out)):
        for j, s in enumerate(STEPS):
            if s in rows:
                v = rows[s]["pgd"]
                ax.bar(i + (j - 1) * w, v - 1, w * 0.9, bottom=1, color=color, alpha=SHADE[s])
                ax.text(i + (j - 1) * w, max(v, 1) + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            else:
                ax.text(i + (j - 1) * w, 1.01, "not\nrun", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    style(ax)
    ax.set_xticks(range(len(groups(out))), [g[0] for g in groups(out)], fontsize=8)
    ax.axhline(1, color=INK, lw=0.8)
    ax.set(ylim=(0.9, 1.72), ylabel="adversarial ratio to the unchanged model\n(1 = as faithful as the unchanged model)")
    ax.set_title("Every placement passes; the SGD LoRA's sit closer to the unchanged model than the Adam LoRA's\n"
                 "(three bars per group: steps 10, 100, 1000, light to dark)", fontsize=9.5)
    save(fig, "log_faithful.png")


def fig_behaviour(out: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.26
    for i, (op, _) in enumerate(OPS):
        rows = {r["step"]: r for r in out["rows"] if r["lora"] == "sgd" and r["op"] == op}
        for j, s in enumerate(STEPS):
            v = rows[s]["p_target"]
            ax.bar(i + (j - 1) * w, v, w * 0.9, color=COLOR["sgd"], alpha=SHADE[s], label=f"step {s}" if i == 0 else None)
            ax.text(i + (j - 1) * w, v + 0.01, f"{round(v, 3):.2f}", ha="center", va="bottom", fontsize=7.5)
    style(ax)
    ax.set_xticks(range(len(OPS)), [label for _, label in OPS], fontsize=8.5)
    ax.set(ylim=(0, 1.1), ylabel="held-out probability of the target")
    ax.set_title("The adjoint split keeps less of the SGD LoRA's behaviour as training goes on", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    save(fig, "log_behaviour.png")


def main() -> None:
    out = load(HERE / "results" / "lora_sgd.json")
    fig_outside(out, load(E14), load(E22))
    fig_readfactor(out)
    fig_faithful(out)
    fig_behaviour(out)


if __name__ == "__main__":
    main()
