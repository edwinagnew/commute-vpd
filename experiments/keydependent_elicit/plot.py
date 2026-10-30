"""E54's figures, one per prediction, from results/stage2.json (no model), with E12's eight copy and cipher states on the
same axes as references (r from E46's stage0b.json, S from E52's baseline_neurons.json). States under their gain floor
are not drawn (their S and r were not read). Each share reads 0 for the original model and 1 for the fine-tuned model.

  figures/p1_route_split.png  r per state: E12 copy and cipher, then elicit and the cipher at matched size (P1)
  figures/p2_steering.png     S per state, the same layout, the 0.3 bound drawn (P2)
  figures/p3_size.png         the cipher's r against ||delta-W||: at the matched step and at its own E12 step, per seed;
                              the elicit states' norms marked, the band P3's 0.1 around the own-step r (P3)

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/keydependent_elicit/plot.py [stage2.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG, EXP = HERE / "figures", HERE.parent
COLOR = {"copy": "#2a78d6", "cipher": "#eb6834", "elicit": "#1f9e6e"}
INK, GRID = "#52514e", "#e4e3df"
E12 = [f"{t}_{s}" for t in ("ident32", "ident32b", "rand32", "rand32b") for s in ("s0", "s1")]


def load(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    st = json.loads(path.read_text())
    e46 = json.loads((EXP / "regating_table" / "results" / "stage0b.json").read_text())["states"]
    e52 = json.loads((EXP / "baseline_neurons" / "results" / "baseline_neurons.json").read_text())["b3"]
    ref = {n: {"r": e46[n.replace("_s", "/s")]["r"], "S": e52[n]["S"]} for n in E12}
    return st, ref


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def strip(ax: Any, st: dict[str, Any], ref: dict[str, dict[str, float]], metric: str) -> None:
    """One column per group; E12's states as references, E54's states filled (matched cipher) or hollow (own step)."""
    groups = [("E12 copy", "copy", [ref[n][metric] for n in E12[:4]], True),
              ("E12 cipher", "cipher", [ref[n][metric] for n in E12[4:]], True),
              ("elicit", "elicit", [st["states"][f"elicit_s{s}"].get(metric) for s in (0, 1)], True),
              ("cipher,\nmatched size", "cipher", [st["states"][f"cipher_s{s}"].get(metric) for s in (0, 1)], True),
              ("cipher,\nown step", "cipher", [st["states"][f"cipher_own_s{s}"].get(metric) for s in (0, 1)], False)]
    for x, (_, kind, vals, filled) in enumerate(groups):
        ys = [v for v in vals if isinstance(v, float)]
        xs = [x + 0.08 * (i - (len(ys) - 1) / 2) for i in range(len(ys))]
        ax.scatter(xs, ys, s=36, color=COLOR[kind] if filled else "white", edgecolors=COLOR[kind], lw=1.4, zorder=3)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
    ax.axvline(1.5, color=GRID, lw=1)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(1, color=INK, lw=0.7, ls="--")


def p1(st: dict[str, Any], ref: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    strip(ax, st, ref, "r")
    v = st["score"]["P1"]
    style(ax, f"P1: re-gating share r of the least-squares split state ({v['status']})", "r")
    fig.tight_layout()
    fig.savefig(FIG / "p1_route_split.png", dpi=150)


def p2(st: dict[str, Any], ref: dict[str, dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    strip(ax, st, ref, "S")
    ax.axhline(0.3, color=COLOR["elicit"], lw=0.8, ls=":")
    ax.annotate("P2 bound 0.3", (1, 0.3), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                ha="right", fontsize=7, color=INK)
    style(ax, f"P2: steering share S ({st['score']['P2']['status']})", "S")
    fig.tight_layout()
    fig.savefig(FIG / "p2_steering.png", dpi=150)


def p3(st: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ft, S = st["finetunes"], st["states"]
    for s, mk in ((0, "o"), (1, "s")):
        m, o = f"cipher_s{s}", f"cipher_own_s{s}"
        for n, filled in ((m, True), (o, False)):
            if isinstance(S[n].get("r"), float):
                ax.scatter(ft[n]["norm"], S[n]["r"], marker=mk, s=40, color=COLOR["cipher"] if filled else "white",
                           edgecolors=COLOR["cipher"], lw=1.4, zorder=3, label=f"seed {s}, {'matched' if filled else 'own'} step {ft[n]['step']}")
        ax.axvline(ft[f"elicit_s{s}"]["norm"], color=COLOR["elicit"], lw=0.8, ls=":" if s else "--")
        if isinstance(S[o].get("r"), float):  # P3's tolerance around the own-step r
            ax.axhspan(S[o]["r"] - 0.1, S[o]["r"] + 0.1, color=COLOR["cipher"], alpha=0.07, lw=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("||delta-W|| (sum of Frobenius norms over the 24 matrices); green lines: elicit states", color=INK, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    style(ax, f"P3: the cipher's r at matched size and at its own step ({st['score']['P3']['status']})", "r")
    fig.tight_layout()
    fig.savefig(FIG / "p3_size.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    st, ref = load(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "stage2.json")
    p1(st, ref)
    p2(st, ref)
    p3(st)


if __name__ == "__main__":
    main()
