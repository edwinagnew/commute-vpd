"""E60's figures from results/table.json, results/stage1.json and results/predictions.json, no model. Every share reads 0
when none of the fine-tune's gain is carried and 1 when all of it is. Colours as E54 and E56: built to reuse blue, built
to be new orange, the elicit map (label open) green.

  figures/table_shares.png  the 18 runs' stand-in ratio, r and S, by task family, built-to-reuse vs built-to-be-new
  figures/gaps.png          per family and reading: lowest built-to-reuse run minus highest built-to-be-new run, beside
                            the larger seed range of the two kinds
  figures/lowercase_S.png   S on E53's six states against the filed expected ranges, with E46's and E54's S as references

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/regating_truth_table/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
FIG, RES = HERE / "figures", HERE / "results"
COLOR = {"reuse": "#2a78d6", "new": "#eb6834", "open": "#1f9e6e"}
INK, GRID = "#52514e", "#e4e3df"
READINGS = [("ratio", "stand-in ratio"), ("r", "re-gating share r"), ("S", "steering share S")]
# (family label, high member: (kind, tasks), low member: (kind, tasks)); the low member is always built to be new
FAMILIES = [("E46\ncopy vs\nrandom map", ("reuse", ["e46/ident32", "e46/ident32b"]), ("new", ["e46/rand32", "e46/rand32b"])),
            ("E53\nlowercase vs\nciphers", ("reuse", ["e53/lowercase"]), ("new", ["e53/cipher", "e53/broad_cipher"])),
            ("E54\nelicit map\nvs cipher", ("open", ["e54/elicit"]), ("new", ["e54/cipher"]))]
LEGEND = {"reuse": "built to reuse (copy, lowercase)", "new": "built to be new (random maps, ciphers)",
          "open": "elicit map: label open"}


def runs() -> dict[str, dict[str, Any]]:
    return json.loads((RES / "table.json").read_text())["runs"]


def values(R: dict[str, dict[str, Any]], tasks: list[str], q: str) -> list[float]:
    return [R[f"{t}_s{s}"][q]["value"] for t in tasks for s in (0, 1)]


def style(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def table_shares(R: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ticks, labels, x, seen = [], [], 0.0, set()
    for qi, (q, _) in enumerate(READINGS):
        for fam, hi, lo in FAMILIES:
            for dx, (kind, tasks) in ((-0.15, hi), (0.15, lo)):
                v = values(R, tasks, q)
                ax.scatter([x + dx + 0.04 * ((i % 4) - 1.5) for i in range(len(v))], v, s=26, color=COLOR[kind],
                           edgecolors="white", lw=0.5, zorder=3, label=LEGEND[kind] if kind not in seen else None)
                seen.add(kind)
            ticks.append(x)
            labels.append(fam)
            x += 1
        if qi < 2:
            ax.axvline(x - 0.5, color=INK, lw=0.6)
    for qi, (_, name) in enumerate(READINGS):
        ax.annotate(name, (qi * 3 + 1, 1.1), ha="center", fontsize=9, color=INK, weight="bold")
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(1, color=INK, lw=0.7, ls="--")
    ax.axhline(0.5, color=INK, lw=0.7, ls=":")
    for y, text in ((0.0, "none of the gain carried: 0"), (1.0, "all of it: 1"),
                    (0.5, "the plan's label cut-off, 0.5 (not filed)")):
        ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=7, color=INK)
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_ylim(-0.05, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    style(ax, "On each task family, does each reading put the runs built to reuse above the runs built to be new?",
          "share of the fine-tune's gain\n(ratio, r: log-odds; S: log P)")
    fig.tight_layout()
    fig.savefig(FIG / "table_shares.png", dpi=150)


def gap_and_range(R: dict[str, dict[str, Any]], hi: list[str], lo: list[str], q: str) -> tuple[float, float]:
    """(min of the high member - max of the low member, larger within-member range): E51's separation quantities."""
    h, low = values(R, hi, q), values(R, lo, q)
    return min(h) - max(low), max(max(h) - min(h), max(low) - min(low))


def gaps(R: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    shade = {"ratio": 1.0, "r": 0.7, "S": 0.4}
    for fi, (fam, hi, lo) in enumerate(FAMILIES):
        for qi, (q, name) in enumerate(READINGS):
            g, rng = gap_and_range(R, hi[1], lo[1], q)
            xx = fi + 0.26 * (qi - 1)
            ax.bar(xx, g, width=0.24, color=INK, alpha=shade[q], label=name if fi == 0 else None)
            ax.scatter(xx, rng, marker="_", s=180, color=COLOR["new"], lw=2, zorder=3,
                       label="larger seed range within a kind" if fi == 0 and qi == 0 else None)
            ax.annotate(f"{g:.2f}", (xx, g), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7,
                        color=INK)
    ax.axhline(0, color=INK, lw=0.7)
    ax.annotate("0 or below: the two kinds overlap", (0, 0), xycoords=("axes fraction", "data"), xytext=(2, -9),
                textcoords="offset points", ha="left", fontsize=7, color=INK)
    ax.set_xticks(range(3), [f[0] for f in FAMILIES], fontsize=8)
    ax.set_ylim(-0.05, 0.9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    style(ax, "How far apart are the two kinds, task family by task family?",
          "lowest run of the higher kind minus highest\nrun of the new kind (S in log P units)")
    fig.tight_layout()
    fig.savefig(FIG / "gaps.png", dpi=150)


def lowercase_S(R: dict[str, dict[str, Any]]) -> None:
    P = json.loads((RES / "predictions.json").read_text())["stage1"]["expected"]
    s1 = json.loads((RES / "stage1.json").read_text())["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for lo_, hi_, kind, text in ((min(values(R, ["e46/ident32", "e46/ident32b"], "S")),
                                  max(values(R, ["e46/ident32", "e46/ident32b"], "S")), "reuse",
                                  "earlier copy task (E46), four runs"),
                                 (min(values(R, ["e46/rand32", "e46/rand32b"], "S")),
                                  max(values(R, ["e46/rand32", "e46/rand32b"], "S")), "new",
                                  "earlier random maps (E46), four runs"),
                                 (min(values(R, ["e54/elicit"], "S")), max(values(R, ["e54/elicit"], "S")), "open",
                                  "elicit map (E54), two runs")):
        ax.axhspan(lo_, hi_, color=COLOR[kind], alpha=0.15, lw=0)
        ax.annotate(text, (1, hi_), xycoords=("axes fraction", "data"), xytext=(-2, 2), textcoords="offset points",
                    ha="right", fontsize=7, color=COLOR[kind])
    names = [("lowercase", "lowercase", "reuse"), ("cipher", "cipher, 50 words", "new"),
             ("broad_cipher", "broad cipher, 500 words", "new")]
    for i, (t, label, kind) in enumerate(names):
        lo_, hi_ = P[t]["range"]
        ax.add_patch(Rectangle((i - 0.3, lo_), 0.6, hi_ - lo_, fill=False, ec=INK, ls="--", lw=0.8,
                                   label="filed expected range" if i == 0 else None))
        ax.scatter(i, P[t]["central"], marker="x", color=INK, s=30, label="filed central value" if i == 0 else None)
        for s in (0, 1):
            ax.scatter(i - 0.08 + 0.16 * s, s1[f"e53/{t}_s{s}"]["S"], s=40, color=COLOR[kind], zorder=3,
                       edgecolors="white", lw=0.5, label=f"measured, two seeds ({LEGEND[kind].split(' (')[0]})"
                       if s == 0 and i < 2 else None)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(3), [n[1] for n in names], fontsize=8)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, frameon=False, loc="upper center")
    style(ax, "Where did steering read on E53's text tasks, against what we filed?",
          "steering share S (share of the log P gain)")
    fig.tight_layout()
    fig.savefig(FIG / "lowercase_S.png", dpi=150)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    R = runs()
    table_shares(R)
    gaps(R)
    lowercase_S(R)


if __name__ == "__main__":
    main()
