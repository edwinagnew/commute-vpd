"""Figures for regating_tasks (E28) from results/regating_tasks.json (JSON only; no model).

One figure per kill condition of H' (reports/decisions/2026-10-22_decision_working-hypothesis-and-two-lines.md), the seven new
states beside E18's three seed-0 readings (recorded in the run's `refs.E18_seed0`, read from E18's JSONs), each with
the kill line and the range H' predicts:
  fixed.png      the updated model's mean trigger values net of W0's (R_mean_net), as a share of each window's own
                 values; E18's filed shared pattern (R_shared) as open markers for context.
  shuffled.png   the fixed change shuffled among each matrix's alive subcomponents (R_shuffled), mean and min-max
                 over 20 draws; the same change unshuffled (R_shared) as open markers.
  other.png      another window's updated values (R_other), mean over 5 derangements, with its reference (W0's values
                 permuted the same way) as open markers.

Usage (repo root):  uv run python experiments/regating_tasks/plot.py [--results PATH] [--figures DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ARM = {"fixed": "#1baf7a", "shuffled": "#e87ba4", "other": "#2a78d6"}  # one colour per arm across figures
E18_TASKS = ("emoticon", "r1_t534_v369", "r2_t359_v816")
LADDER_TASKS = ("r0_t352_v310", "r1_t588_v452", "r2_t275_v2667", "r0_t588_v320")
KV_TASKS = ("ident32", "rand32", "rand1")
NEW_TASKS = LADDER_TASKS + KV_TASKS
GROUPS = (("E18's three tasks", len(E18_TASKS)), ("new word tasks", len(LADDER_TASKS)),
          ("new key-value tasks (hatched)", len(KV_TASKS)))
KILL = {"fixed": 0.5, "shuffled": 0.2, "other": 0.5}
BAND = {"fixed": (0.7, 0.9), "shuffled": (-0.03, 0.03), "other": (0.85, 1.0)}
TITLE = {"fixed": "Does one fixed change give back at least half of what each input's own values give?",
         "shuffled": "Does the change stop working when its values go to the wrong subcomponents?",
         "other": "Does another input's updated values work about as well as the input's own?"}
YLABEL = {"fixed": "mean values net of W0's mean,\nshare of own values' recovery",
          "shuffled": "shuffled change,\nshare of own values' recovery",
          "other": "another input's values,\nshare of own values' recovery"}
KEY = {"fixed": "R_mean_net", "shuffled": "R_shuffled", "other": "R_other"}


def rows(out: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per plotted state: E18's three (seed 0), then the new states in run order."""
    e18 = out["refs"]["E18_seed0"]
    res = [{"name": n, "src": "E18", "scored": dict.fromkeys(ARM, True), **e18[n]} for n in E18_TASKS]
    kills = out.get("score", {}).get("kills", {})
    for n in NEW_TASKS:
        if n not in out["states"]:
            continue
        m = out["states"][n]
        k = kills.get(n, {})
        res.append({"name": n, "src": "E28", "scored": {a: k.get(a) in ("kill", "survives") for a in ARM}, "m": m,
                    **{key: m[key] for key in ("R_mean_net", "R_shared", "R_shuffled", "R_other", "R_other_W0")}})
    return res


def label(r: dict[str, Any], arm: str) -> str:
    return r["name"] if r["src"] == "E18" or r["scored"][arm] else f"{r['name']}\nnot scored"


def style(ax: Any, arm: str, n: int) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=0.8)
    lo, hi = BAND[arm]
    ax.axhspan(lo, hi, color=ARM[arm], alpha=0.12, label=f"range H' predicts ({lo:g} to {hi:g})")
    ax.axhline(KILL[arm], color=INK, ls="--", lw=1.2, label=f"kill line ({KILL[arm]:g})")
    start = 0
    for name, size in GROUPS:
        if start:
            ax.axvline(start - 0.5, color=MUTED, lw=0.8)
        ax.text(start + (size - 1) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=MUTED, fontsize=8)
        start += size
    ax.set_xlim(-0.6, n - 0.4)


def fig(rs: list[dict[str, Any]], arm: str, figdir: Path) -> None:
    f, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(rs))
    style(ax, arm, len(rs))
    vals = [r[KEY[arm]] for r in rs]
    alpha = [0.45 if r["src"] == "E18" else (1.0 if r["scored"][arm] else 0.25) for r in rs]
    for i, (v, a) in enumerate(zip(vals, alpha, strict=True)):
        kv = rs[i]["name"] in KV_TASKS
        ax.bar(i, v, 0.6, color=ARM[arm], alpha=a, hatch="//" if kv else None, edgecolor="white" if kv else None,
               label=YLABEL[arm].split(",")[0] if i == 0 else None)
        text = f"{v:+.3f}" if arm == "shuffled" else f"{v:.2f}"
        ax.text(i + 0.32, v, text, ha="left", va="center", color=INK, fontsize=7.5)
    if arm == "fixed":
        ax.scatter(x, [r["R_shared"] for r in rs], marker="o", facecolors="none", edgecolors=INK, zorder=3,
                   label="E18's filed shared pattern, clip(g0 + s) (context, not scored)")
        new = [(i, r["m"]["rho"]) for i, r in enumerate(rs) if "m" in r]
        ax.scatter([i for i, _ in new], [rho["mean_W0"] / rho["trig"] for _, rho in new], marker="x", color=INK,
                   zorder=3, label="subtracted reference: W0's mean values (new states)")
    if arm == "other":
        ax.scatter(x, [r["R_other_W0"] for r in rs], marker="o", facecolors="none", edgecolors=INK, zorder=3,
                   label="reference: W0's values, permuted the same way")
    if arm == "shuffled":
        ax.scatter(x, [r["R_shared"] for r in rs], marker="o", facecolors="none", edgecolors=INK, zorder=3,
                   label="the same change unshuffled, clip(g0 + s)")
        for i, r in enumerate(rs):
            if "m" in r:
                s = r["m"]["shuffled"]
                ax.vlines(i, s["R_min"], s["R_max"], color=INK, lw=1.0, label="min to max over 20 draws" if i == len(E18_TASKS) else "_nolegend_")
    ax.set_xticks(x, [label(r, arm) for r in rs], fontsize=7.5)
    ax.set_ylabel(YLABEL[arm])
    ax.set_title(TITLE[arm], fontsize=10.5, pad=20)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    f.tight_layout()
    f.savefig(figdir / f"{arm}.png", dpi=150)
    plt.close(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results" / "regating_tasks.json")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    if not args.results.is_file():
        sys.exit(f"no results yet: {args.results} (run run.py first)")
    out = json.loads(args.results.read_text())
    rs = rows(out)
    args.figures.mkdir(parents=True, exist_ok=True)
    for arm in ARM:
        fig(rs, arm, args.figures)
    print(f"wrote {', '.join(f'{a}.png' for a in ARM)} to {args.figures}")


if __name__ == "__main__":
    main()
