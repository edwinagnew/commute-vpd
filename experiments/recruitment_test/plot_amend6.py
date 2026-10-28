"""Figures for E38 Amendment 6, S and D restored together and S split by direction, from its results (no model), and
the few derived numbers the analysis quotes.

  amend6_union.png  per task-seed: the restored share rho_r(X) = (T1 - lo(W1, X's trigger outputs put back to W0's)) / G
                    of S, D and S union D, against what the union would read if the two sets carried the same part
                    (the larger set's share, plus P14's 0.05 'redundant' margin) and fully separate parts with no
                    saturation (the sum); P14's 'separate' bar (larger + half the smaller); Amendment 4's every alive
                    subcomponent restored together for scale.
  amend6_split.png  per task-seed: rho_r of S, of S's target-pointing members S+ (training e_c > 0) and of its other
                    members S-, beside D, with the first-order prediction e-hat (held-out e_c summed / G) of S+ and S-
                    as hollow markers and the filed reference lines on emoticon (D - 0.05 for S+, -0.05 for S-).

Derived and printed, not filed: the first-order prediction of D2, the next S-count per matrix by training e_c outside S
and D, the set the untested control would restore with D (top_per_matrix, as Amendment 5 picked D).

Reads results/amend6.json, results/amend5.json, results/large/amend5_scores.pt and
experiments/transport_test/results/large/alive.pt.
Usage (repo root):  ~/commute-agents/fleet/cpu-run uv run python experiments/recruitment_test/plot_amend6.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import torch
from plot import BLUE, HERE, INK, MUTED, ORANGE, RES, VIOLET, keys, save, style
from plot_amend2 import short_labels

from commute import recruitment as RT

ALIVE = HERE.parent / "transport_test" / "results" / "large" / "alive.pt"
SCORES = RES / "large" / "amend5_scores.pt"
LIGHT_BLUE, RED = "#8fbcef", "#e34948"  # S+ (S's target-pointing members), S- (its other members)


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def shade_we(ax: Any, y: float) -> None:
    ax.axvspan(3.5, 5.5, color="#d9d8d4", alpha=0.45, lw=0)
    ax.text(4.5, y, "' we': S below 0.10,\nnot scored", ha="center", fontsize=7, color=MUTED)


def fig_union(st: dict[str, Any], a5: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    shade_we(ax, 0.66)
    for i, k in enumerate(ks):
        r, first = st[k]["rho"], i == 0
        hi, lo = max(r["S"], r["D"]), min(r["S"], r["D"])
        ax.fill_between([i - 0.36, i + 0.36], [hi] * 2, [hi + 0.05] * 2, color=MUTED, alpha=0.25, lw=0, zorder=1,
                        label="same part: the larger set's share, plus the 0.05 margin" if first else None)
        ax.plot([i - 0.36, i + 0.36], [hi + 0.5 * lo] * 2, color=INK, ls=":", lw=1.2, zorder=2,
                label="'separate' bar: the larger share plus half the smaller" if first else None)
        ax.plot([i - 0.36, i + 0.36], [r["S"] + r["D"]] * 2, color=VIOLET, ls="--", lw=1.4, zorder=2,
                label="separate parts, adding exactly: S's share plus D's" if first else None)
        ax.plot([i - 0.4, i + 0.4], [a5["states"][k]["rho_all_alive_amend4"]] * 2, color=INK, lw=1.8, zorder=2,
                label="every alive subcomponent restored together (Amendment 4)" if first else None)
        for j, (x, col, mk, lab) in enumerate((("S", BLUE, "o", "switched-on set S"),
                                               ("D", ORANGE, "v", "direction-matched set D"),
                                               ("S_union_D", VIOLET, "D", "S and D restored together"))):
            ax.scatter([i + (j - 1) * 0.17], [r[x]], s=58, marker=mk, color=col, zorder=4, label=lab if first else None)
            ax.annotate(f"{r[x]:.2f}", (i + (j - 1) * 0.17 + 0.05, r[x]), fontsize=6.5, color=INK, va="center")
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("share of the fine-tune's gain taken back when the set's\ntrigger outputs are put back to the original model's")
    ax.set_title("Restored together, do S and D take back the same part of the gain, or two parts?\n"
                 "S and D are disjoint, each with S's count in every matrix", fontsize=10)
    style(ax, zero=False)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    save(fig, "amend6_union.png")


def fig_split(st: dict[str, Any]) -> None:
    ks = keys()
    sets = (("S", BLUE, "o", "switched-on set S"),
            ("S_plus", LIGHT_BLUE, "^", "S+: S's members whose changes point toward the target"),
            ("S_minus", RED, "v", "S-: S's other members"),
            ("D", ORANGE, "s", "direction-matched set D, for reference"))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    shade_we(ax, 0.45)
    for i, k in enumerate(ks):
        r, h, first = st[k]["rho"], st[k]["e_hat"], i == 0
        for j, (x, col, mk, lab) in enumerate(sets):
            xi = i + (j - 1.5) * 0.17
            ax.scatter([xi], [r[x]], s=54, marker=mk, color=col, zorder=4, label=lab if first else None)
            ax.annotate(f"{r[x]:.2f}", (xi + 0.04, r[x]), fontsize=6, color=INK, va="center")
            if x in h:
                ax.scatter([xi], [h[x]], s=54, marker=mk, facecolors="none", edgecolors=col, zorder=3,
                           label="first-order prediction for S+ and S-, hollow" if first and x == "S_plus" else None)
        if k.startswith("emoticon"):
            ax.plot([i - 0.4, i + 0.4], [r["D"] - 0.05] * 2, color=ORANGE, ls=":", lw=1.2, zorder=2,
                    label="emoticon, filed for S+: at least D's share minus 0.05" if first else None)
            ax.plot([i - 0.4, i + 0.4], [-0.05] * 2, color=RED, ls=":", lw=1.2, zorder=2,
                    label="emoticon, filed for S-: at most -0.05" if first else None)
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(-0.25, 0.52)
    ax.set_ylabel("share of the fine-tune's gain taken back when the set's\ntrigger outputs are put back to the original model's")
    ax.set_title("Does the switched-on set's target-pointing half take back what the direction-matched set does?\n"
                 "S split by the sign of each member's first-order take-back on the training windows", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    save(fig, "amend6_split.png")


def next_set(a5s: dict[str, Any], sc: dict[str, Any], alive: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """D2: per matrix, S's count of alive pieces outside S and D with the largest training e_c, as D was picked."""
    m = a5s["members"]
    sel = {x: {k: torch.tensor(v, dtype=torch.long) for k, v in m[x].items()} for x in ("S", "D")}
    excl = {k: torch.cat([sel["S"][k], sel["D"][k]]) for k in sel["S"]}
    d2, short = RT.top_per_matrix(sc["e_train"], alive, excl, {k: int(v.numel()) for k, v in sel["S"].items()})
    assert not any(short.values())
    return d2


def derived(st: dict[str, Any], a5: dict[str, Any]) -> None:
    alive = torch.load(ALIVE, weights_only=False)["alive"]
    scores = torch.load(SCORES, weights_only=True)
    for k in keys():
        r, h, G, a5s = st[k]["rho"], st[k]["e_hat"], st[k]["gain"], a5["states"][k]
        h5 = a5s["e_hat"]
        d2 = next_set(a5s, scores[k], alive)
        e2 = RT.summed(scores[k]["e_held"], d2) / G
        pos2 = sum(int((scores[k]["e_train"][m][v] > 0).sum()) for m, v in d2.items())
        a = st[k]["a"]
        print(f"{k}: G {G:.3f} | rho S {r['S']:.4f} D {r['D']:.4f} U {r['S_union_D']:.4f} sum {r['S'] + r['D']:.4f} "
              f"U/sum {r['S_union_D'] / (r['S'] + r['D']):.3f} | a {a:.4f} bar {0.5 * min(r['S'], r['D']):.4f} "
              f"a/min {a / min(r['S'], r['D']):.3f} | all alive {a5s['rho_all_alive_amend4']:.4f} | S+ {r['S_plus']:.4f} "
              f"S- {r['S_minus']:.4f} S+ + S- {r['S_plus'] + r['S_minus']:.4f} S+ - D {r['S_plus'] - r['D']:+.4f} | "
              f"e-hat S+ {h['S_plus']:.4f} (/rho {h['S_plus'] / r['S_plus']:.2f}) S- {h['S_minus']:.4f} | "
              f"e-hat S {h5['S']:.4f} D {h5['D']:.4f} S+D {h5['S'] + h5['D']:.4f} | "
              f"D2 n {sum(int(v.numel()) for v in d2.values())} pos {pos2} e-hat {e2:.4f} e-hat D+D2 {h5['D'] + e2:.4f}")


def main() -> None:
    a6, a5 = load("amend6.json"), load("amend5.json")
    st = a6["states"]
    fig_union(st, a5)
    fig_split(st)
    derived(st, a5)
    print("P14", json.dumps(a6["P14"]))
    print("reported", json.dumps(a6["reported"]))


if __name__ == "__main__":
    main()
