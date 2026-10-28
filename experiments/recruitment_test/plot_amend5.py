"""Figures for E38 Amendment 5, the direction-matched restore, from its results (no model), and the few derived numbers
the analysis quotes.

  amend5_restore.png    per task-seed: the restored share rho_r(X) = (T1 - lo(W1, X's trigger outputs put back to W0's))
                        / G of the switched-on set S, the direction-matched set D (per matrix, S's count of the alive
                        pieces outside S with the largest first-order take-back e_c) and the size-matched set C_k (the
                        same count by Amendment 4's output change), with zero, S's level minus P13's 0.05 margin, the
                        0.10 floor S needs for a task to be scored, and Amendment 4's every-alive-piece share for scale.
  amend5_linearity.png  does the first-order take-back predict the restore? left: R6's single pieces, e_c against the
                        measured a_c (nats, training windows); right: e-hat(X) = sum of held-out e_c over X / G against
                        rho_r(X) for S, D, C_k and every alive piece; each panel with its y = x line.
  amend5_topdir.png     per task-seed: S's top-direction share against the filed 0.5 and what a random set of S's size
                        per matrix would read.

Reads results/amend5.json, results/amend4.json and experiments/transport_test/results/large/alive.pt (alive counts per
matrix, for the chance level only).
Usage (repo root):  ~/commute-agents/fleet/cpu-run uv run python experiments/recruitment_test/plot_amend5.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from plot import BLUE, HERE, INK, MUTED, ORANGE, RES, keys, save, style
from plot_amend2 import short_labels
from plot_amend4 import AQUA

ALIVE = HERE.parent / "transport_test" / "results" / "large" / "alive.pt"
SETS = (("S", BLUE, "o", "switched-on set S"),
        ("D", ORANGE, "v", "direction-matched D: S's count per matrix, largest first-order take-back, outside S"),
        ("C_k", AQUA, "s", "size-matched C_k: S's count per matrix, largest output change, outside S"))


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def chance_top_dir(sel: dict[str, int], alive: dict[str, int]) -> float:
    """S's expected top-direction share if S were a random set of its size in each matrix."""
    return sum(n * n / alive[k] for k, n in sel.items()) / sum(sel.values())


def fig_restore(st: dict[str, Any]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axvspan(3.5, 5.5, color="#d9d8d4", alpha=0.45, lw=0)
    ax.text(4.5, 0.505, "' we': S below the floor,\ntask not scored", ha="center", fontsize=7, color=MUTED)
    for i, k in enumerate(ks):
        r, first = st[k]["rho"], i == 0
        ax.plot([i - 0.38, i + 0.38], [st[k]["rho_all_alive_amend4"]] * 2, color=INK, lw=1.6, zorder=2,
                label="every alive subcomponent restored together (Amendment 4)" if first else None)
        ax.plot([i - 0.3, i + 0.3], [r["S"] - 0.05] * 2, color=BLUE, lw=1, ls=":", zorder=2,
                label="S's share minus 0.05: D below this would read 'S ahead'" if first else None)
        for j, (x, col, mk, lab) in enumerate(SETS):
            ax.scatter([i + (j - 1) * 0.16], [r[x]], s=58, marker=mk, color=col, zorder=4, label=lab if first else None)
            ax.annotate(f"{r[x]:.2f}", (i + (j - 1) * 0.16 + 0.05, r[x]), fontsize=6.5, color=INK, va="center")
    ax.axhline(0.10, color=INK, ls="--", lw=1, label="floor: S must reach 0.10 on both seeds for its task to be scored")
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.set_ylim(-0.2, 0.55)
    ax.set_ylabel("share of the fine-tune's gain taken back when the set's\ntrigger outputs are put back to the original model's")
    ax.set_title("Does S take back more of the gain than the subcomponents whose changes point most toward the target?\n"
                 "all three sets hold S's count in every matrix", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    save(fig, "amend5_restore.png")


def fig_linearity(a5: dict[str, Any]) -> None:
    st = a5["states"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.0))
    for t, (task, col) in enumerate((("emoticon", BLUE), ("r1_t534_v369", ORANGE), ("r2_t359_v816", AQUA))):
        pts = [p for name, v in a5["controls"][task]["R6"].items() for p in v["pieces"]]
        ax.scatter([p[2] for p in pts], [p[3] for p in pts], s=16, color=col, zorder=3,
                   label=short_labels()[2 * t].split("\n")[0])
    lim = (-2.4, 0.8)
    ax.plot(lim, lim, color=MUTED, lw=1, label="measured = predicted")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted, first order: e_c, nats")
    ax.set_ylabel("measured: drop in the target's log-odds when\nthis one subcomponent is restored, nats")
    ax.set_title("Single subcomponents: the 10 largest predictions per state", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    for x, col, mk, lab in (*SETS, ("all_alive", INK, "X", "every alive subcomponent")):
        pred = [st[k]["e_hat"][x] for k in keys()]
        meas = [st[k]["rho"][x] if x != "all_alive" else st[k]["rho_all_alive_amend4"] for k in keys()]
        bx.scatter(pred, meas, s=40, marker=mk, color=col, zorder=3, label=lab.split(":")[0])
    k = "emoticon/s0"
    bx.annotate("emoticon, both seeds:\npredicted negative, measured +0.38",
                (st[k]["e_hat"]["all_alive"], st[k]["rho_all_alive_amend4"]),
                xytext=(-20, 12), textcoords="offset points", fontsize=6.5, color=INK)
    lim = (-0.7, 0.6)
    g = np.linspace(0.05, lim[1], 20)
    bx.fill_between(g, g / 2, np.minimum(2 * g, lim[1]), color="#d9d8d4", alpha=0.5, lw=0,
                    label="within a factor 2, the filed expectation for S and D")
    bx.plot(lim, lim, color=MUTED, lw=1, label="measured = predicted")
    bx.axhline(0, color=MUTED, lw=0.6)
    bx.axvline(0, color=MUTED, lw=0.6)
    bx.set_xlim(lim)
    bx.set_ylim(lim)
    bx.set_xlabel("predicted, first order: summed e_c over the set, as a share of the gain")
    bx.set_ylabel("measured share of the gain taken back")
    bx.set_title("Sets, six task-seed states each", fontsize=9)
    bx.legend(frameon=False, fontsize=7, loc="lower right")
    for a in (ax, bx):
        style(a, zero=False)
    fig.suptitle("Does the first-order take-back predict what restoring a subcomponent, or a set, actually takes back?",
                 fontsize=10)
    save(fig, "amend5_linearity.png")


def fig_topdir(st: dict[str, Any], alive: dict[str, int]) -> None:
    ks = keys()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, k in enumerate(ks):
        s = st[k]["sets"]
        ax.bar(i, s["top_dir_share"], 0.4, color=BLUE, label="measured" if i == 0 else None)
        ax.text(i, s["top_dir_share"] + 0.012, f"{s['top_dir_share']:.2f}", ha="center", fontsize=7)
        ax.plot([i - 0.3, i + 0.3], [chance_top_dir(s["S_by_matrix"], alive)] * 2, color=MUTED, lw=2,
                label="what a random set of S's size per matrix would read" if i == 0 else None)
    ax.axhline(0.5, color=INK, ls="--", lw=1, label="filed expectation: below 0.5 on emoticon and ' which'")
    ax.set_xticks(range(len(ks)), short_labels(), fontsize=8)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("share of S among the top S-count\nsubcomponents per matrix by first-order take-back")
    ax.set_title("Is the switched-on set made of the subcomponents whose changes point most toward the target?",
                 fontsize=10)
    style(ax, zero=False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, "amend5_topdir.png")


def main() -> None:
    a5 = load("amend5.json")
    st = a5["states"]
    alive = {k: int(v.sum()) for k, v in torch.load(ALIVE, weights_only=False)["alive"].items()}
    fig_restore(st)
    fig_linearity(a5)
    fig_topdir(st, alive)
    for k in keys():  # the derived numbers the analysis quotes
        r, h, G, s = st[k]["rho"], st[k]["e_hat"], st[k]["gain"], st[k]["sets"]
        print(f"{k}: G {G:.3f} | rho S {r['S']:+.4f} ({r['S'] * G:+.2f} nats) D {r['D']:+.4f} ({r['D'] * G:+.2f}) "
              f"C_k {r['C_k']:+.4f} | D - S {r['D'] - r['S']:+.4f} | e-hat/rho S {h['S'] / r['S']:.3f} D "
              f"{h['D'] / r['D']:.3f} C_k {h['C_k'] / r['C_k']:.3f} all {h['all_alive'] / st[k]['rho_all_alive_amend4']:+.3f} "
              f"| top-dir {s['top_dir_share']:.4f} chance {chance_top_dir(s['S_by_matrix'], alive):.4f}")
    for c in a5["controls"].values():
        for name, v in c.get("R6", {}).items():
            e, a = np.array([p[2] for p in v["pieces"]]), np.array([p[3] for p in v["pieces"]])
            print(f"R6 {name}: agree {v['agree']} | n e<0 {(e < 0).sum()} | a/e median {np.median(a / e):.3f} "
                  f"min {(a / e).min():+.3f} | top piece {v['pieces'][0][:2]}")
    print("P13", json.dumps(a5["P13"]))


if __name__ == "__main__":
    main()
