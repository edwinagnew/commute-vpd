"""E312 (P2.84) figure: the reverse fine-tune's half-life per Fig 4 task, each state beside its rotated null, with D
(the distance it undoes) on the same x below, and the controls on the right. Reads results/summary.json; `--stub`
draws made-up numbers into figures/sketch_p2-84.png (the card's sketch). Censored readings sit at the cap with an
open marker; voids are left out and counted in the title.

    ~/commute-agents/fleet/cpu-run python3 experiments/reverse_ft_fig4/plot.py [--stub]
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
STATE, NULL, GAIN, INK = "#257de0", "#999999", "#f47230", "#191919"
CAP = 1000


def stub() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    groups = {}
    for i, g in enumerate(["c3/rank1", "c3/cipher", "c3/copy", "e46/ident32", "e46/rand32", "e53/lowercase", "e53/cipher", "e54/elicit",
                           "e54/cipher", "e54/cipher_own", "e58/rank1", "e58/cipher", "e58/copy", "e58/prior_cipher", "e62/copy", "e62/lookup"]):  # fmt: skip
        t = 8 + 6 * rng.random()
        groups[g] = {"states": {f"{g}_s{s}": {"state": {"t_half": t + rng.normal(0, 0.5), "D": 5 + rng.random()},
                                               "null": {"t_half": "censored" if i % 5 == 0 else 30 + 20 * rng.random(), "D": 0.3 * rng.random()}}
                                for s in range(3)}}  # fmt: skip
    gain = {f"gain/{t}_s{s}": {"t_half": 3 + rng.random(), "D": 1 + rng.random()} for t in ("copy", "rank1", "elicit", "prior_cipher") for s in range(3)}
    planted = {"wrapper": {"t_half": 3.97, "D": 4.21}, "partners": {f"c3/cipher_s{s}": {"t_half": 11 + rng.random(), "D": 4.21} for s in range(3)}}
    return {"groups": groups, "gain": gain, "planted": planted, "gate": {"reading": "pass (made up)"}, "D_void": 1e-9}


def val(r: dict[str, Any] | None, key: str) -> tuple[float | None, bool]:
    """(value, censored); None for a missing or void reading."""
    if r is None or r.get(key) is None:
        return None, False
    return (float(CAP), True) if r[key] == "censored" else (float(r[key]), False)


def dots(ax: Any, x: float, r: dict[str, Any] | None, key: str, color: str, marker: str) -> int:
    v, cens = val(r, key)
    if v is None:
        return 1
    ax.scatter([x], [v], s=36, marker=marker, facecolors="none" if cens else color, edgecolors=color, linewidths=1.5, zorder=3)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    S = stub() if a.stub else json.loads((HERE / "results" / "summary.json").read_text())
    names = list(S["groups"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex="col", gridspec_kw={"width_ratios": [4, 1], "height_ratios": [3, 2]})
    (top, ctl), (bot, ctl_d) = axes
    void = 0
    for i, g in enumerate(names):
        for j, v in enumerate(S["groups"][g]["states"].values()):
            x = i + (j - 1) * 0.12
            void += dots(top, x - 0.03, v["state"], "t_half", STATE, "o") + dots(top, x + 0.03, v["null"], "t_half", NULL, "s")
            dots(bot, x - 0.03, v["state"], "D", STATE, "o")
            dots(bot, x + 0.03, v["null"], "D", NULL, "s")
    pl = S["planted"]
    ctl_items = [("planted\nwrapper", [pl["wrapper"]], STATE), ("scaled\ncipher", list(pl["partners"].values()), NULL), ("gain-only", list(S["gain"].values()), GAIN)]
    for i, (_lab, rs, col) in enumerate(ctl_items):
        for j, r in enumerate(rs):
            x = i + (j - len(rs) / 2) * 0.04
            dots(ctl, x, r, "t_half", col, "o")
            dots(ctl_d, x, r, "D", col, "o")
    for ax in (top, ctl):
        ax.set_yscale("log")
        ax.axhline(CAP, color=NULL, lw=0.8, ls=":")
    top.set_ylabel("steps to undo half (t_half)")
    bot.set_ylabel("D = KL(M || F) at the read, nats")
    bot.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ctl_d.set_xticks(range(3), [c[0] for c in ctl_items], fontsize=8)
    top.set_title(f"Reverse fine-tune per Fig 4 task: states (blue) beside their norm-matched rotated nulls (grey); open = censored at {CAP}; {void} void", fontsize=9, color=INK)
    ctl.set_title(f"Controls: gate {S['gate']['reading']}", fontsize=9, color=INK)
    for ax in axes.flat:
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    note = ("Each null is the state's update turned by a random rotation per matrix: matched on per-matrix norm (spectrum), not on D; "
            "t_half rises as D shrinks (C3 ladder), so a null's longer t_half is partly a size effect. D is drawn below each pair.\n"
            "C3 (E55): on the c3/* states this port undid the old-by-construction rank-1 map more slowly than the new cipher (13.3 vs 10.1 steps). "
            f"Items with D under D_void {S['D_void']:.2g} (10 x the measured kappa floor) are void and not drawn.")  # fmt: skip
    fig.text(0.01, 0.005, note, fontsize=7.5, color="#666666")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = HERE / "figures" / ("sketch_p2-84.png" if a.stub else "reverse_ft_fig4.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
