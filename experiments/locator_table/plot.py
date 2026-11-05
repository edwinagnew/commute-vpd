"""E97's figures, one per claim, from results/*.json; no model.

  figures/plants.png        carry share of every part on the two planted edits (stage 0's known answers)
  figures/copy_lookup.png   removal, carry and the edge to the logits: copy against lookup, every part
  figures/additivity.png    single edges summed against the same edges patched together, E62's six states
  figures/cross_faith.png   share of a state's own gap kept under the original model's circuit
  figures/lowercase.png     the pair (1,1)+(1,2) off: the state's fall against the original model's, both units

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/locator_table/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"


def load(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


S0, S1, S2, A0, A1 = (load(f) for f in ("stage0.json", "stage1.json", "stage2.json", "amend1_stage0.json", "amend1_stage1.json"))
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1f8a6d", "#7a4fb5"
INK, MUTED, GRID, BAND, DRAW = "#2b2a28", "#52514e", "#e4e3df", "#ecebe6", "#8f8e88"
SURFACE = "#fcfcfb"
PARTS = [p for b in range(4) for p in [*(f"{b},{h}" for h in range(6)), f"mlp{b}"]]
SEEDS = (0, 1, 2)
A = float(S1["a"])  # the located bar, read from copy in stage 1
CONTROL_MAX = 0.2


def f(x: Any) -> float:
    return float(x)  # the JSON writes nan as the string "nan"


def axes(ax: Any, xlabel: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)


def title(ax: Any, text: str) -> None:
    ax.set_title(text, color=INK, loc="left", fontsize=11, pad=10)


def save(fig: Any, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def state(js: dict[str, Any], task: str, s: int) -> dict[str, Any]:
    return js["states"][js["index"][task][str(s)]]


# ----------------------------------------------------------------------------- figure 1: the plants


def plants() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = range(len(PARTS))
    for k, (name, col, dx) in enumerate((("block3", BLUE, -0.2), ("block1", ORANGE, 0.2))):
        carry = S0["plants"][name]["reads"]["carry"]
        lab = {"block3": "edit planted in block 3's MLP", "block1": "edit planted in block 1's MLP"}[name]
        ax.bar([i + dx for i in x], [f(carry[p]["lo"]) for p in PARTS], width=0.4, color=col, label=lab)
    floor = f(S0["plants"]["block1"]["inheritance"])
    ax.axhline(1.0, color=DRAW, ls="--", lw=1)
    ax.text(len(PARTS) - 0.5, 1.02, "expected at the edited MLP: 1, exactly", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.axhspan(-floor, floor, color=BAND, zorder=0)
    ax.text(0, floor + 0.02, f"largest share passed on downstream of the block-1 edit: {floor:.3f}", color=MUTED, fontsize=8)
    ax.set_xticks(list(x), [p if p.startswith("mlp") else f"({p})" for p in PARTS], rotation=90, fontsize=7)
    axes(ax, "part: head (block, head) or a block's MLP, in model order", "carry share: gain lost when the part's output is W₀'s")
    title(ax, "Does the carry read put a planted edit where it was made, and nothing upstream?")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    save(fig, "plants.png")


# ----------------------------------------------------------------------------- figure 2: copy against lookup


def removal(js: dict[str, Any], t: str, s: int, p: str) -> float:
    return f(state(js, t, s)["removal"][p]["share"]["lo"])


def carry(t: str, s: int, p: str) -> float:
    return f(state(S1, t, s)["carry"][p]["lo"])


def direct(t: str, s: int, p: str) -> float:
    g = state(A1, t, s)["gain"]
    return f(g["to_logits"][p]) / f(g["G"])


def spread(read: Any, t: str, p: str) -> tuple[float, float, float]:
    v = [read(t, s, p) for s in SEEDS]
    return sum(v) / len(v), min(v), max(v)


def copy_lookup() -> None:
    reads = (("Removal: the part at W₀'s mean", lambda t, s, p: removal(S1, t, s, p), PARTS + ["2,3+2,4"]),
             ("Carry: the part's output from W₀", carry, PARTS),
             ("Edge: the part's direct write to the logits", direct, PARTS))  # fmt: skip
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, (name, read, parts) in zip(axs, reads):
        ax.add_patch(Rectangle((A, -0.2), 2, CONTROL_MAX + 0.2, color=BAND, zorder=0))
        ax.text(A + 0.02, -0.15, f"located: copy ≥ {A}\nlookup ≤ {CONTROL_MAX}", color=MUTED, fontsize=8)
        ax.plot([-0.2, 1.4], [-0.2, 1.4], color=GRID, lw=1)
        for p in parts:
            (cx, cl, ch), (lx, ll, lh) = spread(read, "T5 copy", p), spread(read, "T5 lookup", p)
            hi = p in ("2,4", "mlp3", "2,3+2,4")
            col = BLUE if p in ("2,4", "2,3+2,4") else ORANGE if p == "mlp3" else DRAW
            ax.errorbar(cx, lx, xerr=[[cx - cl], [ch - cx]], yerr=[[lx - ll], [lh - lx]], fmt="o", ms=7 if hi else 4,
                        color=col, ecolor=col, elinewidth=0.8, zorder=3 if hi else 2)  # fmt: skip
            if hi or abs(cx) > 0.5 or abs(lx) > 0.5:
                lab = p if p.startswith("mlp") else f"({p.replace('+', ')+(')})"
                ax.annotate(lab, (cx, lx), xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK if hi else MUTED)
        axes(ax, "share of the copy state's gain, mean of 3 seeds", "share of the lookup state's gain" if ax is axs[0] else "")
        title(ax, name)
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-0.2, 1.1)
    fig.suptitle("Which read names the copying head (2,4) and spares the lookup? Every part, bars span the seeds", color=INK, x=0.01, ha="left")
    save(fig, "copy_lookup.png")


# ----------------------------------------------------------------------------- figure 3: additivity


def additivity() -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    tol = f(load("predictions_amend1.json")["settings"]["edge_sum_tol"])
    ax.fill_between([-0.1, 1.6], [-0.1 - tol, 1.6 - tol], [-0.1 + tol, 1.6 + tol], color=BAND, zorder=0)
    ax.plot([-0.1, 1.6], [-0.1, 1.6], color=DRAW, ls="--", lw=1)
    ax.text(1.0, 0.82, f"filed tolerance: ±{tol} of the gain", color=MUTED, fontsize=8, rotation=38)
    for t, col in (("T5 copy", BLUE), ("T5 lookup", ORANGE)):
        for s in SEEDS:
            r = state(A1, t, s)
            joint, sums = r["gain"]["joint_share"], r["check"]["sum_share"]
            for p in PARTS:
                x, y = f(joint[p]), f(sums[p])
                mlp = p.startswith("mlp")
                ax.scatter(x, y, s=40 if mlp else 14, color=col, marker="s" if mlp else "o", alpha=0.9 if mlp else 0.5,
                           label=f"{t[3:]} states" if (s, p) == (0, "mlp0") else None)  # fmt: skip
                if (t, s, p) == ("T5 copy", 0, "mlp2"):
                    ax.annotate("mlp2, copy seeds 0 and 2:\noutside the tolerance", (x, y), xytext=(10, -28), textcoords="offset points", fontsize=8, color=INK)
    pl = A0["plants"]["block1"]
    G = f(pl["gain"]["G"])
    ax.scatter(f(pl["gain"]["joint_share"]["mlp1"]), f(pl["gain"]["out_sum"]["mlp1"]) / G, s=60, marker="D", color=PURPLE,
               label="block-1 planted edit, its MLP")  # fmt: skip
    axes(ax, "share of the gain lost: every edge out of the part patched together", "the same edges patched one at a time, shares summed")
    title(ax, "Do the part's single-edge shares add up to the part?\nSquares: MLPs (the stopping check); dots: heads")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylim(-0.1, 1.6)
    save(fig, "additivity.png")


# ----------------------------------------------------------------------------- figure 4: cross-faithfulness


def cross_faith() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = [("T5 copy", s, "lookup") for s in SEEDS] + [("T5 lookup", s, "copy") for s in SEEDS]
    for i, (t, s, other) in enumerate(rows):
        c = state(A1, t, s)["cross"]
        own, oth = f(c["own"]["faith"]), f(c[other]["faith"])
        lab = {0: "copy state on its own pair (filed)", 3: "lookup state on its own pair (filed)"}.get(i)
        ax.scatter(i, own, s=70, color=BLUE if t == "T5 copy" else ORANGE, zorder=3, label=lab)
        if math.isfinite(oth):
            ax.scatter(i, oth, s=70, facecolor="none", edgecolor=MUTED, lw=1.5, zorder=3, label="the other map's pair, under W₀'s circuit for it (beside)" if i == 0 else None)
        else:
            ax.text(i, 1.75, "no gap:\nnot read", ha="center", color=MUTED, fontsize=8)
    w0 = A1["w0"]["copy"]
    w0f = f(w0["faith"][str(w0["n"])])
    ax.axhline(w0f, color=GREEN, lw=1.2, label=f"W₀ itself on the copy pair, its own circuit: {w0f:.2f}")
    ax.axhline(0.85, color=DRAW, ls=":", lw=1, label="Hanna et al.'s circuit bar: 0.85")
    ax.axhspan(0.32, 0.98, color=BAND, zorder=0)
    ax.text(-0.4, 0.62, "between the band's ends:\ncopy ≥ 0.98, lookup ≤ 0.32", color=MUTED, fontsize=8)
    ax.set_xticks(range(6), [f"{t[3:]} seed {s}" for t, s, _ in rows])
    axes(ax, "", "share of the state's clean-minus-corrupted gap kept")
    ax.set_ylim(0, 2.05)
    title(ax, "Does the copying circuit of the original model W₀ carry each state's own task?")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "cross_faith.png")


# ----------------------------------------------------------------------------- figure 5: lowercase in both units


def lowercase() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    pair = "1,1+1,2"
    units = (("lo", "log-odds of the target against all tokens (filed unit)"), ("gap", "log p(lowercase) − log p(capitalised)"))
    for ax, (u, uname) in zip(axs, units):
        groups = [("lowercase seed 0", "T7 lowercase", 0), ("lowercase seed 1", "T7 lowercase", 1), ("cipher seed 0", "T8 cipher", 0), ("cipher seed 1", "T8 cipher", 1)]
        for i, (lab, t, s) in enumerate(groups):
            r = state(S2, t, s)
            rm = r["removal"][pair]
            vals = (f(r["G"][u]), f(rm["fall"][u]), f(rm["w0_move"][u]))
            for k, (v, col) in enumerate(zip(vals, (DRAW, BLUE, GREEN))):
                ax.bar(i + (k - 1) * 0.27, v, width=0.27, color=col,
                       label=("gain: state minus W₀", "the state's fall with the pair off", "W₀'s fall with the pair off")[k] if i == 0 else None)  # fmt: skip
            ax.text(i, max(vals) + 0.08, f"{vals[1] / vals[0]:.2f} of gain", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(4), [g[0] for g in groups], fontsize=8)
        ax.axhline(0, color=GRID)
        axes(ax, "", "nats" if ax is axs[0] else "")
        title(ax, uname)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("With heads (1,1) and (1,2) at W₀'s mean, does the lowercase state lose its gain, or a part both models need?", color=INK, x=0.01, ha="left")
    save(fig, "lowercase.png")


if __name__ == "__main__":
    plants()
    copy_lookup()
    additivity()
    cross_faith()
    lowercase()
