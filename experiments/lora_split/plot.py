"""Figures for lora_split from results/lora_split.json (or the smoke file with --smoke).

  blocks.png      one question: which side of the subcomponents does each update miss? Per update (fine-tune, 24-matrix
                  LoRA), the four blocks against the step as stacked bars, with the random-update null as a hatched bar
                  beside each; the non-saturated pooling (the scored one).
  sides.png       one question: on which side is each update inside? Read-side-inside share / null and write-side-inside
                  share / null against the step for both updates, with the one-matrix LoRAs as points.
  outside.png     one question: how far inside is each update, and does rescaling ever explain it? Outside-both share /
                  null and rescaling share / null against the step for both updates, with the one-matrix LoRAs of
                  ranks 1 to 32 at their site as points at their steps.
  pile.png        one question: on one card, which map survives the Pile adversary at matched steps? PGD ratio of
                  do-nothing, least-squares split and scaled adjoint split for both updates at the part-2 steps, with
                  the W0 floor band; the adjoint state's held-out P(o) against the update's beside it.

Usage (repo root):  uv run python experiments/lora_split/plot.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLOCKS = ("in_in", "in_out", "out_in", "out_out")
LABELS = {"in_in": "inside both", "in_out": "write side only", "out_in": "read side only", "out_out": "outside both"}
COLORS = {"in_in": "#2a78d6", "in_out": "#8fbce6", "out_in": "#f2b48c", "out_out": "#eb6834"}
UPDATES = {"ft": "full fine-tune", "lora24": "rank-8 LoRA, 24 matrices"}
GREY = "#7f7f7f"


def load(smoke: bool) -> dict[str, Any]:
    return json.loads((HERE / "results" / f"lora_split{'_smoke' if smoke else ''}.json").read_text())


def pooled(out: dict[str, Any], u: str, s: str) -> dict[str, float]:
    rec = out["table"][u][s]
    return rec["pooled_ns"] or rec["pooled_all"]


def blocks(out: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (u, title) in zip(axes, UPDATES.items()):
        steps = sorted(out["table"][u], key=int)
        x = range(len(steps))
        for i, s in enumerate(steps):
            p = pooled(out, u, s)
            base = base_null = 0.0
            for b in BLOCKS:
                ax.bar(i - 0.2, p[b], 0.4, bottom=base, color=COLORS[b], label=LABELS[b] if i == 0 else None)
                ax.bar(i + 0.2, p["null_" + b], 0.4, bottom=base_null, color=COLORS[b], hatch="//", edgecolor="white", alpha=0.6)
                base, base_null = base + p[b], base_null + p["null_" + b]
        ax.set_xticks(list(x), steps)
        ax.set_xlabel("step")
        ax.set_title(f"{title}: left bar; random update of its size: hatched", fontsize=10)
    axes[0].set_ylabel("share of input-weighted energy")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ratio(p: dict[str, float], field: str) -> float:
    if field in ("read_in", "write_in"):
        a, b = ("in_in", "out_in") if field == "read_in" else ("in_in", "in_out")
        return (p[a] + p[b]) / (p["null_" + a] + p["null_" + b])
    null = "rescaling_null" if field == "rescaling" else "null_" + field
    return p[field] / p[null]


def ratios(out: dict[str, Any], path: Path, panels: tuple[tuple[str, str], ...], ylim: tuple[float, float] | None) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4), sharey=ylim is not None)
    for ax, (field, title) in zip(axes, panels):
        for u, label in UPDATES.items():
            steps = sorted(out["table"][u], key=int)
            ax.plot([int(s) for s in steps], [ratio(pooled(out, u, s), field) for s in steps], "o-", label=label)
        for u in sorted(k for k in out["table"] if k.startswith("site")):
            steps = sorted(out["table"][u], key=int)
            ax.plot([int(s) for s in steps], [ratio(out["table"][u][s]["pooled_all"], field) for s in steps],
                    "x", color=GREY, label="one-matrix LoRAs (ranks 1-32)" if u.endswith("r1_s0") else None)
        ax.axhline(1, color=GREY, linewidth=1, linestyle="--", label="a random update reads 1")
        ax.set_xscale("log")
        ax.set_xlabel("step")
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sides(out: dict[str, Any], path: Path) -> None:
    ratios(out, path, (("read_in", "read side: energy on what the subcomponents read / its null"),
                       ("write_in", "write side: energy in what the subcomponents write / its null")), (0.6, 1.8))


def outside(out: dict[str, Any], path: Path) -> None:
    ratios(out, path, (("out_out", "outside both / its null"), ("rescaling", "rescaling share / its null")), None)


def pile(out: dict[str, Any], path: Path) -> None:
    part2 = out.get("part2") or {}
    if not part2 or out["W0"]["pgd_floor"] is None:
        return
    line = 1 + 3 * out["W0"]["pgd_floor"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for u, label in UPDATES.items():
        steps = sorted(part2[u], key=int)
        xs = [int(s) for s in steps]
        for arm, name, style in (("nothing", "do-nothing", ":"), ("lift", "least-squares split", "-"), ("adjoint", "adjoint split, scaled", "--")):
            axes[0].plot(xs, [part2[u][s][arm]["pile"]["pgd_kl"] for s in steps], "o" + style, label=f"{label}, {name}")
        axes[1].plot(xs, [part2[u][s]["adjoint"]["routes"]["p_cells"]["T1"] for s in steps], "o--", label=f"{label}, adjoint state")
        axes[1].plot(xs, [part2[u][s]["adjoint"]["p_o_update"] for s in steps], "o:", label=f"{label}, the update")
    axes[0].axhspan(1, line, color=GREY, alpha=0.2, label="W0 floor (3 seeds)")
    axes[0].set_yscale("log")
    axes[0].set_title("Pile adversarial faithfulness ratio to W0, three states per update")
    axes[1].set_title("held-out P(target) at the trigger, adjoint state vs its update")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = load(args.smoke)
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    blocks(out, figs / f"blocks{suffix}.png")
    sides(out, figs / f"sides{suffix}.png")
    outside(out, figs / f"outside{suffix}.png")
    pile(out, figs / f"pile{suffix}.png")
    print(f"wrote {figs}")


if __name__ == "__main__":
    main()
