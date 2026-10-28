"""Figures for E38 Amendment 7, the settling control for Amendment 6's 'separate', from its results (no model), and the
few derived numbers the analysis quotes.

  amend7_cover.png  per task-seed: the restored share rho_r(X) = (T1 - lo(W1, X's trigger outputs put back to W0's)) / G
                    of T2 (the top 2 n_k per matrix by training e_c, S's members allowed) against S union D, the same
                    count; P16's 'short' line (S union D - 0.05); the first-order prediction e-hat (held-out e_c summed
                    / G) of both as hollow markers; Amendment 4's every alive subcomponent restored together for scale.
  amend7_pair.png   per task-seed: rho_r of D, D2 (the next n_k per matrix outside S union D) and D union D2, against what
                    the union would read if the two carried the same part (the larger share, plus the 0.05 margin) and
                    separate parts adding exactly (the sum); P15's 'separate' bar (larger + half the smaller); the
                    0.10 floor each set needs to be scored.
  amend7_line.png   the 67M restore line in one figure, per task-seed: random sets (Amendment 3, mean and range), the
                    pieces outside S whose trigger outputs changed most at S's count (Amendment 4), S, D (Amendment 5),
                    S union D (Amendment 6) and T2; every alive subcomponent for scale.

Derived and printed, not filed: union over sum for (D, D2) beside Amendment 6's (S, D); a2's margin over its bar; how T2
is made (all of D in it, how many of S and of D2); rho_r(T2) - rho_r(S union D) beside its first-order prediction.

Reads results/amend7.json, amend6.json, amend5.json, amend4.json, amend3.json.
Usage (repo root):  ~/commute-agents/fleet/cpu-run uv run python experiments/recruitment_test/plot_amend7.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from plot import BLUE, INK, MUTED, ORANGE, RES, VIOLET, keys, save, style
from plot_amend2 import short_labels

LIGHT_ORANGE, RUST, TEAL, GOLD = "#f5a97f", "#a8431c", "#0f7b6c", "#b8860b"  # D2, D union D2, T2, size-ranked set
YLABEL = "share of the fine-tune's gain taken back when the set's\ntrigger outputs are put back to the original model's"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def bar(ax: Any, i: float, y: float, w: float = 0.4, **kw: Any) -> None:
    ax.plot([i - w, i + w], [y, y], **kw)


def dot(ax: Any, x: float, y: float, col: str, mk: str, lab: str | None, hollow: bool = False) -> None:
    kw = {"facecolors": "none", "edgecolors": col} if hollow else {"color": col}
    ax.scatter([x], [y], s=58, marker=mk, zorder=4 if not hollow else 3, label=lab, **kw)
    if not hollow:
        ax.annotate(f"{y:.2f}", (x + 0.05, y), fontsize=6.5, color=INK, va="center")


def finish(ax: Any, title: str, name: str, ylim: tuple[float, float], zero: bool = True) -> None:
    ks = keys()
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(YLABEL)
    ax.set_title(title, fontsize=10)
    style(ax, zero=zero)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    save(ax.figure, name)


def fig_cover(st: dict[str, Any], a5: dict[str, Any]) -> None:
    _fig, ax = plt.subplots(figsize=(10, 5.8))
    for i, k in enumerate(keys()):
        r, h, first = st[k]["rho"], st[k]["e_hat"], i == 0
        bar(ax, i, a5["states"][k]["rho_all_alive_amend4"], color=INK, lw=1.8, zorder=2,
            label="every alive subcomponent restored together (Amendment 4)" if first else None)
        bar(ax, i, r["S_union_D"] - 0.05, w=0.3, color=VIOLET, ls=":", lw=1.2, zorder=2,
            label="'short' line: S and D together, minus 0.05" if first else None)
        dot(ax, i - 0.12, r["S_union_D"], VIOLET, "D", "S and D restored together (Amendment 6)" if first else None)
        dot(ax, i + 0.12, r["T2"], TEAL, "s", "T2: top twice S's count by pull on the target, S allowed" if first else None)
        dot(ax, i - 0.12, h["S_union_D"], VIOLET, "D", None, hollow=True)
        dot(ax, i + 0.12, h["T2"], TEAL, "s", "first-order prediction, hollow" if first else None, hollow=True)
        frac = st[k]["sets"]["T2_holds_of_S_frac"]
        ax.text(i, 0.03, f"T2 holds\n{frac:.0%} of S", ha="center", fontsize=6.5, color=MUTED)
    finish(ax, "Does one ranking by pull on the target, at the same count, take back what S and D take back together?\n"
           "Both sets hold twice S's count in every matrix", "amend7_cover.png", (0, 1.2))


def fig_pair(st: dict[str, Any], a6: dict[str, Any]) -> None:
    _fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axvspan(1.5, 5.5, color="#d9d8d4", alpha=0.45, lw=0)
    ax.text(3.5, 0.62, "D2 below 0.10 on ' which' and ' we': not scored", ha="center", fontsize=7, color=MUTED)
    ax.axhline(0.10, color=MUTED, ls="-.", lw=0.9, zorder=1, label="0.10: the share each set needs to be scored")
    for i, k in enumerate(keys()):
        r, first = st[k]["rho"], i == 0
        hi, lo = max(r["D"], r["D2"]), min(r["D"], r["D2"])
        ax.fill_between([i - 0.36, i + 0.36], [hi] * 2, [hi + 0.05] * 2, color=MUTED, alpha=0.25, lw=0, zorder=1,
                        label="same part: the larger set's share, plus the 0.05 margin" if first else None)
        bar(ax, i, hi + 0.5 * lo, w=0.36, color=INK, ls=":", lw=1.2, zorder=2,
            label="'separate' bar: the larger share plus half the smaller" if first else None)
        bar(ax, i, r["D"] + r["D2"], w=0.36, color=RUST, ls="--", lw=1.4, zorder=2,
            label="separate parts, adding exactly: D's share plus D2's" if first else None)
        for j, (x, col, mk, lab) in enumerate((("D", ORANGE, "v", "direction-matched set D"),
                                               ("D2", LIGHT_ORANGE, "^", "D2: the next-ranked set, same count"),
                                               ("D_union_D2", RUST, "D", "D and D2 restored together"))):
            dot(ax, i + (j - 1) * 0.17, r[x], col, mk, lab if first else None)
        r6 = a6["states"][k]["rho"]
        ax.text(i, 0.69, f"together/summed\n{r['D_union_D2'] / (r['D'] + r['D2']):.2f}\n"
                f"S and D: {r6['S_union_D'] / (r6['S'] + r6['D']):.2f}", ha="center", fontsize=6.5, color=MUTED)
    finish(ax, "Do two disjoint target-ranked sets, D and the next-ranked D2, also take back about the sum of their parts?\n"
           "Each holds S's count in every matrix; neither holds any of S's members", "amend7_pair.png", (0, 0.8),
           zero=False)


def fig_line(st: dict[str, Any], a3: dict[str, Any], a4: dict[str, Any], a5: dict[str, Any]) -> None:
    _fig, ax = plt.subplots(figsize=(10.5, 6))
    sets = ((-0.30, GOLD, "P", "outside S, largest change in trigger output, S's count (Amendment 4)"),
            (-0.15, BLUE, "o", "switched-on set S (Amendment 3)"),
            (0.0, ORANGE, "v", "outside S, largest pull on the target, S's count: D (Amendment 5)"),
            (0.15, VIOLET, "D", "S and D together, twice S's count (Amendment 6)"),
            (0.30, TEAL, "s", "top by pull on the target, S allowed, twice S's count: T2"))
    for i, k in enumerate(keys()):
        first = i == 0
        rnd = a3["states"][k]["rho"]["random"]
        ax.vlines(i - 0.42, min(rnd), max(rnd), color=MUTED, lw=1.4, zorder=2,
                  label="20 random sets of S's size: range and mean (Amendment 3)" if first else None)
        ax.scatter([i - 0.42], [sum(rnd) / len(rnd)], s=22, color=MUTED, zorder=3)
        bar(ax, i, a5["states"][k]["rho_all_alive_amend4"], w=0.46, color=INK, lw=1.6, zorder=2,
            label="every alive subcomponent restored together (Amendment 4)" if first else None)
        ys = (a4["states"][k]["rho"]["C_n"], st[k]["rho"]["S"], st[k]["rho"]["D"], st[k]["rho"]["S_union_D"],
              st[k]["rho"]["T2"])
        for (dx, col, mk, lab), y in zip(sets, ys, strict=True):
            ax.scatter([i + dx], [y], s=50, marker=mk, color=col, zorder=4, label=lab if first else None)
    finish(ax, "What does each restored set take back? Γ's switched-on set beats random and size-ranked sets;\n"
           "sets ranked by pull on the target, at the same count, match or beat it", "amend7_line.png", (-0.3, 0.75))


def derived(st: dict[str, Any], a6: dict[str, Any], a5: dict[str, Any], a7: dict[str, Any]) -> None:
    for k in keys():
        r, h, s = st[k]["rho"], st[k]["e_hat"], st[k]["sets"]
        r6 = a6["states"][k]["rho"]
        m, d = st[k]["members"], a5["states"][k]["members"]["D"]
        d_out = sum(len(set(v) - set(m["T2"].get(mk, []))) for mk, v in d.items())
        d2_in = sum(len(set(v) & set(m["T2"].get(mk, []))) for mk, v in m["D2"].items())
        lo = min(r["D"], r["D2"])
        print(f"{k}: G {st[k]['gain']:.3f} | D {r['D']:.4f} D2 {r['D2']:.4f} DuD2 {r['D_union_D2']:.4f} "
              f"sum {r['D'] + r['D2']:.4f} U/sum {r['D_union_D2'] / (r['D'] + r['D2']):.3f} "
              f"(S,D {r6['S_union_D'] / (r6['S'] + r6['D']):.3f}) | a2 {st[k]['a2']:.4f} bar {0.5 * lo:.4f} "
              f"margin {st[k]['a2'] - 0.5 * lo:+.4f} a2/min {st[k]['a2'] / lo:.3f} | T2 {r['T2']:.4f} SuD "
              f"{r['S_union_D']:.4f} diff {r['T2'] - r['S_union_D']:+.4f} first-order diff "
              f"{h['T2'] - h['S_union_D']:+.4f} e-hat/rho T2 {h['T2'] / r['T2']:.2f} SuD {h['S_union_D'] / r['S_union_D']:.2f}"
              f" | T2 = D (missing {d_out}) + {s['T2_holds_of_S']} of S + {d2_in} of D2; n {st[k]['n']['T2']}")
    print("P15", json.dumps(a7["P15"]))
    print("P16", json.dumps(a7["P16"]))
    print("reported", json.dumps(a7["reported"]))


def main() -> None:
    a7, a6, a5, a4, a3 = (load(f"amend{n}.json") for n in (7, 6, 5, 4, 3))
    st = a7["states"]
    fig_cover(st, a5)
    fig_pair(st, a6)
    fig_line(st, a3, a4, a5)
    derived(st, a6, a5, a7)


if __name__ == "__main__":
    main()
