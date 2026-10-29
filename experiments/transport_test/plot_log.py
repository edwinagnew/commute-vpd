"""The research log's three figures for E1, redrawn from saved results with plain labels (no model).

  log_span.png       per fine-tune and step: the share of the update's squared size (Frobenius) that lies in the span
                     of the alive subcomponents, against the share a random direction of the same per-matrix sizes
                     would have (sum over matrices of its share of the update times alive / (d_out d_in)).
                     Reads results/transport_{A,B}.json (T0_frobenius) and calibration.json (alive per matrix).
  log_behaviour.png  emoticon fine-tune, step 10: the share of the fine-tune's effect at the emoticon position that
                     each map reproduces on held-out windows (capture): the least-squares rescaling split, a fitted
                     rescaling of every alive subcomponent, and free write-vector edits on the 3 or 10 subcomponents
                     with the largest least-squares contribution against 3 or 10 random ones (5 seeds).
                     Reads results/transport_A.json (T0_frobenius_capture, T0_functional, T1).
  log_location.png   emoticon fine-tune: the rank of the VPD paper's emoticon subcomponent (h.2.mlp.down_proj:2359) and
                     its five siblings among the 358 alive subcomponents of that matrix, by least-squares contribution,
                     at each step. Reads results/transport_A.json (1b_location).

Usage (repo root):  python experiments/transport_test/plot_log.py
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from plot import AQUA, BLUE, FIGURES, INK, MUTED, ORANGE, load, save

STEPS = (10, 30, 100, 300)
ARMS = (("A", "emoticon fine-tune", BLUE), ("B", "continued pretraining", AQUA))
D_MODEL, D_MLP = 768, 3072


def need(name: str) -> dict[str, Any]:
    d = load(name)
    assert d is not None, f"missing results/{name}"
    return d


def random_share(norm2: dict[str, float], alive: dict[str, int]) -> float:
    """Expected span share of a random direction with the update's per-matrix sizes."""
    tot = sum(norm2.values())
    size = {m: D_MODEL * (D_MLP if "mlp" in m else D_MODEL) for m in norm2}
    return sum(v / tot * alive[m] / size[m] for m, v in norm2.items())


def fig_span(tr: dict[str, dict[str, Any]], alive: dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for arm, lab, col in ARMS:
        fr = [tr[arm]["steps"][str(s)]["T0_frobenius"] for s in STEPS]
        ys = [f["fraction_alive_pooled"] / random_share(f["dW_norm2_per_matrix"], alive) for f in fr]
        ax.plot(STEPS, ys, "-o", color=col, lw=2, label=lab)
        for s, y, f in zip(STEPS, ys, fr, strict=True):
            txt = f"{100 * f['fraction_alive_pooled']:.2f}%"
            if arm == "A":
                ax.text(s, y + 0.35, txt, color=col, ha="center", fontsize=7.5)
            elif s == STEPS[0]:
                ax.text(s, y - 0.55, txt, color=col, ha="center", fontsize=7.5)
            elif s == STEPS[-1]:
                ax.text(s * 1.07, y, txt, color=col, va="center", fontsize=7.5)
    ax.axhline(1, color=MUTED, lw=1.2)
    ax.text(STEPS[0], 0.3, "a random direction (about 0.035% of it in the span)", color=MUTED, fontsize=8.5)
    ax.axhline(10, color=ORANGE, lw=1.2, ls="--")
    ax.text(STEPS[0], 10.25, "predicted before the run for continued pretraining: ten times random", color=ORANGE, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.set_xlim(8, 400)
    ax.set_ylim(0, 11.5)
    ax.set_xlabel("fine-tune step")
    ax.set_ylabel("share of the update inside the span of the\nexisting pieces, in multiples of a random direction's")
    ax.set_title("Over 99.6% of both updates lies outside the existing pieces (labels: share inside);\n"
                 "only the emoticon update drifts toward them as it keeps training", fontsize=10)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    save(fig, "log_span.png")


def fig_behaviour(a: dict[str, Any]) -> None:
    r = a["steps"]["10"]
    t1 = r["T1"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    single = (("rescaling split\n(least squares)", r["T0_frobenius_capture"]["primary"]["capture"]),
              ("rescaling refitted\non behaviour,\nall 9,881 pieces", r["T0_functional"]["primary"]["capture"]))
    for i, (lab, v) in enumerate(single):
        ax.bar(i, v, 0.6, color=INK if i == 0 else AQUA)
        ax.text(i, v + 0.02, f"{v:.3f}" if v < 0.01 else f"{v:.2f}", ha="center", fontsize=8.5)
    ticks = [lab for lab, _ in single]
    for j, k in enumerate(("3", "10")):
        x = 2 + j
        top = t1[k]["write"]["primary"]["capture"]
        rnd = [z["primary"]["capture"] for z in t1[k]["random_write"]]
        m = sum(rnd) / len(rnd)
        ax.bar(x - 0.17, top, 0.32, color=BLUE, label="the pieces with the largest shares of the rescaling split" if j == 0 else None)
        ax.bar(x + 0.17, m, 0.32, color=ORANGE, alpha=0.85, label="random pieces (bar mean, dots 5 seeds)" if j == 0 else None)
        ax.scatter([x + 0.17] * len(rnd), rnd, s=12, color="k", zorder=3)
        ax.text(x - 0.17, top + 0.02, f"{top:.2f}", ha="center", fontsize=8.5)
        ax.text(x + 0.17, max(rnd) + 0.02, f"{m:.2f}", ha="center", fontsize=8.5)
        ticks.append(f"free write edits\non {k} pieces")
    ax.set_xticks(range(4), ticks, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of the fine-tune's effect at the\nemoticon reproduced on held-out text")
    ax.set_title("The rescaling split carries none of the emoticon behaviour; a full refit or ten edited pieces carry it,\n"
                 "but ten random pieces do as well as the ten chosen ones (step 10, one fine-tune seed)", fontsize=9.5)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "log_behaviour.png")


def fig_location(a: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ranks = {s: a["steps"][str(s)]["1b_location"]["ranks"] for s in STEPS}
    ax.axhspan(0.8, 8.5, color="#eeeae0", zorder=0)
    ax.text(STEPS[0] * 1.15, 1.2, "top eight", color=MUTED, fontsize=8)
    for c in ranks[STEPS[0]]:
        ys = [ranks[s][c] for s in STEPS]
        main = c == "2359"
        ax.plot(STEPS, ys, "-o", color=BLUE if main else MUTED, lw=2.4 if main else 1, ms=6 if main else 3.5,
                alpha=1 if main else 0.7, label="the paper's emoticon piece" if main else None, zorder=3 if main else 2)
        if main:
            for s, y in zip(STEPS, ys, strict=True):
                ax.annotate(str(y), (s, y), textcoords="offset points", xytext=(9, -3), color=BLUE, fontsize=8.5, fontweight="bold")
    ax.plot([], [], "-o", color=MUTED, lw=1, ms=3.5, label="its five siblings")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(400, 0.8)
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.set_yticks([1, 2, 5, 10, 30, 100, 358], ["1", "2", "5", "10", "30", "100", "358 (last)"])
    ax.set_xlabel("fine-tune step")
    ax.set_ylabel("rank in its matrix by share of the\nrescaling split (1 = largest)")
    ax.set_title("Rank of the paper's emoticon piece and its siblings in their matrix (one seed):\nsecond at step 10, between 23rd and near last after", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, "log_location.png")


def main() -> None:
    tr = {arm: need(f"transport_{arm}.json") for arm, _, _ in ARMS}
    FIGURES.mkdir(exist_ok=True)
    fig_span(tr, need("calibration.json")["alive_per_module"])
    fig_behaviour(tr["A"])
    fig_location(tr["A"])


if __name__ == "__main__":
    main()
