"""E79's figures from the results JSONs, and the per-row means from the saved per-position arrays; no model.

  figures/two_by_two.png  forward KL to the fine-tuned model: W0's or A's masks, nothing placed or A's change, with do-nothing on the same axis
  figures/own_split.png   own-mask read against clean + self, for the states near the fine-tuned model
  figures/teeth.png       own-mask read against clean KL, every state read under its masks, with the filed teeth bar
  figures/frozen.png      W0's frozen masks: forward KL by state, on maths and web text, with the references
  figures/agree_soft.png  top-token agreement against entropy, by state and row set
  results/rows.json       per-row forward KL on the maths rows, from results/large/

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_masked_read/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from safetensors.numpy import load_file

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
COLOUR = {"W0": "#57564f", "A": "#1baf7a", "B": "#2a78d6", "L": "#eb6834", "N": "#7b4fc4", "Z": "#c2387f", "P": "#a58a12", "C": "#8d8c85"}
MARK = {"W0": "P", "A": "s", "B": "o", "L": "D", "N": "*", "Z": "X", "P": "^", "C": "v"}
READ, A1, A2 = (json.loads((RES / f"{f}.json").read_text()) for f in ("read", "amend1", "amend2"))
R, O, N = READ["reads"], A1["reads"], A2["reads"]


def fwd(cell: dict) -> float:
    return cell["all"]["fwd"]


def style(ax: Axes, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)  # pyright: ignore[reportArgumentType]
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def finish(fig: Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, facecolor="white", dpi=130)
    plt.close(fig)


def two_by_two() -> None:
    rows = [("W0's masks, nothing placed", fwd(O["W0"]["M"]["d1"]), "#b9b8b1"), ("W0's masks, A's change", fwd(R["A"]["M"]["d1"]), COLOUR["A"]),
            ("A's masks, nothing placed (Z)", fwd(N["Z"]["M"]["d1"]), "#b9b8b1"), ("A's masks, A's change", fwd(O["A"]["M"]["d1"]), COLOUR["A"]),
            ("Do-nothing: the fine-tuned model's weights\nand masks, the update in Δ (N)", fwd(N["N"]["M"]["d1"]), COLOUR["N"])]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ys = [5.2, 4.2, 2.8, 1.8, 0.4]
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.8)
    for y, (_, v, _c) in zip(ys, rows):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(ys, [r[0] for r in rows], color=INK)
    k0 = O["W0"]["M"]["d1"]["all"]["self_fwd"]
    ax.axvline(k0, color=INK, ls=":", lw=1.2)
    ax.text(k0 + 0.012, 6.0, f"W0 masked against W0 clean: {k0:.3f}", color=INK, fontsize=9, va="center")
    ax.set_ylim(-0.3, 6.3)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("forward KL(fine-tuned clean ‖ state masked), 8 maths rows, nats")
    ax.set_title("Which moves the masked read: the masks, or the change placed?", color=INK, x=0.2, fontsize=11)
    style(ax, "x")
    finish(fig, "two_by_two.png")


def own_split() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, rs, title in zip(axes, "MW", ("Maths rows", "Web text rows")):
        states = ["N", "A", "L", "B"]
        clean = [abs(fwd(N["N"][rs]["clean"])) if s == "N" else fwd(O[s][rs]["clean"]) for s in states]
        self_ = [(N if s == "N" else O)[s][rs]["d1"]["all"]["self_fwd"] for s in states]
        own = [fwd((N if s == "N" else O)[s][rs]["d1"]) for s in states]
        xs = range(len(states))
        ax.bar(xs, self_, color="#b9b8b1", width=0.6, label="self: KL(state clean ‖ state masked)")
        ax.bar(xs, clean, bottom=self_, color=INK, width=0.6, label="clean: KL(fine-tuned clean ‖ state clean)")
        ax.scatter(xs, own, color="white", edgecolor=INK, s=60, zorder=3, label="the own-mask read itself")
        k0 = O["W0"][rs]["d1"]["all"]["self_fwd"]
        ax.axhline(k0, color=INK, ls=":", lw=1.2)
        ax.text(3.45, k0 + 0.004, f"W0's self {k0:.3f}", color=INK, fontsize=9, ha="right")
        ax.set_xticks(list(xs), ["N, do-nothing", "A", "L", "B"], color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        style(ax)
    axes[0].set_ylabel("forward KL, nats")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("For states near the fine-tuned model, is the own-mask read anything beyond clean + self?", color=INK, x=0.02, ha="left", fontsize=11)
    finish(fig, "own_split.png")


def teeth() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    src = {s: O[s]["M"] for s in ("W0", "A", "L", "B", "P", "C")} | {s: N[s]["M"] for s in "NZ"}
    place = {"W0": (-6, 9), "P": (-30, -3), "C": (9, -3), "Z": (-58, 9), "N": (9, -11), "A": (-14, 9), "L": (-14, 9), "B": (9, 2)}
    words = {"W0": "W0", "A": "A", "L": "L", "B": "B", "N": "N, do-nothing", "Z": "Z: W0 under A's masks", "P": "P", "C": "C"}
    for s, c in src.items():
        x, y = max(fwd(c["clean"]), 0.0), fwd(c["d1"])
        ax.scatter(x, y, color=COLOUR[s], marker=MARK[s], s=95, zorder=3, edgecolor="white", lw=0.8)
        ax.annotate(words[s], (x, y), textcoords="offset points", xytext=place[s], color=INK, fontsize=9.5)
    bar = 3 * fwd(O["A"]["M"]["d1"])
    ax.axhline(bar, color=INK, ls="--", lw=1)
    ax.text(0.0016, bar + 0.02, f"filed teeth bar, 3 x A's read: {bar:.3f}", color=INK, fontsize=9)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlim(-0.002, 3.5)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("clean KL(fine-tuned clean ‖ state clean), nats, symlog axis, linear under 0.01")
    ax.set_ylabel("own-mask read, forward KL, nats")
    ax.set_title("What did the teeth rule test? Maths rows, each state under its masks", color=INK, loc="left", fontsize=11)
    style(ax, "both")
    finish(fig, "teeth.png")


def frozen() -> None:
    states = ["W0", "A", "L", "B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    refs = {"M": [(READ["score"]["B"]["n5"], "--", "W0 at its best single temperature"), (fwd(N["Z"]["M"]["d1"]), "-.", "Z, nothing placed, A's masks"),
                  (O["W0"]["M"]["d1"]["all"]["self_fwd"], ":", "W0 masked against W0 clean")],
            "W": [(fwd(N["Z"]["W"]["d1"]), "-.", "Z, nothing placed, A's masks"), (O["W0"]["W"]["d1"]["all"]["self_fwd"], ":", "W0 masked against W0 clean")]}
    for ax, rs, title in zip(axes, "MW", ("Maths rows", "Web text rows, note the scale")):
        vals = [fwd(R[s][rs]["d1"]) for s in states]
        ax.bar(range(4), vals, width=0.6, color=[COLOUR[s] for s in states])
        top = max(vals) * 1.25
        for i, v in enumerate(vals):
            ax.text(i, v + top * 0.012, f"{v:.3f}", ha="center", color=INK, fontsize=9.5)
        for v, ls, words in refs[rs]:
            ax.axhline(v, color=INK, ls=ls, lw=1.1, label=f"{words}: {v:.3f}")
        ax.text(4, top * 0.28, "N under\nthese masks:\nnot read\non these rows", ha="center", color=MUTED, fontsize=9)
        ax.set_xticks(range(5), ["W0", "A", "L", "B", "N"], color=INK)
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(0, top)
        ax.set_title(title, color=INK, fontsize=10)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=8, loc="upper right")
        style(ax)
    axes[0].set_ylabel("forward KL(fine-tuned clean ‖ state masked), nats")
    fig.suptitle("Under W0's frozen masks, how far does each state fall?", color=INK, x=0.02, ha="left", fontsize=11)
    finish(fig, "frozen.png")


LABEL = {("M", "W0"): (-14, 9), ("M", "A"): (8, -12), ("M", "L"): (6, -12), ("M", "B"): (-48, -14),
         ("W", "W0"): (4, 9), ("W", "A"): (6, -12), ("W", "L"): (-46, -14), ("W", "B"): (-34, 9)}


def agree_soft() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharey=True)
    for ax, rs, title in zip(axes, "MW", ("Maths rows", "Web text rows")):
        ax.axvline(R["FT"][rs]["clean"]["all"]["H"], color=INK, ls="--", lw=1)
        ax.text(R["FT"][rs]["clean"]["all"]["H"] + 0.02, 0.992, f"fine-tuned clean\nentropy {R['FT'][rs]['clean']['all']['H']:.2f}", color=INK, fontsize=8.5, va="top")
        for s in ("W0", "A", "L", "B"):
            c = R[s][rs]["d1"]["all"]
            ax.scatter(c["H"], c["agree"], color=COLOUR[s], marker=MARK[s], s=85, zorder=3, edgecolor="white", lw=0.8)
            ax.annotate(f"{s} frozen", (c["H"], c["agree"]), textcoords="offset points", xytext=LABEL[rs, s], color=INK, fontsize=8.5)
        for s in "NZ":
            c = N[s][rs]["d1"]["all"]
            ax.scatter(c["H"], c["agree"], facecolor="white", edgecolor=COLOUR[s], marker=MARK[s], s=85, zorder=3, lw=1.3)
            ax.annotate(s, (c["H"], c["agree"]), textcoords="offset points", xytext=(-5, 8), color=INK, fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("mean entropy of the masked output, nats")
        style(ax, "both")
    a0, a1 = READ["score"]["B"]["a0"], READ["score"]["B"]["a1"]
    mid = a0 + (a1 - a0) / 2
    axes[0].axhline(mid, color=INK, ls=":", lw=1.2)
    axes[0].text(0.62, mid + 0.003, f"filed agreement bar {mid:.3f}", color=INK, fontsize=8.5)
    axes[0].set_ylabel("top-token agreement with the fine-tuned model")
    axes[0].set_ylim(0.78, 1.0)
    fig.suptitle("Agreement or softness? Solid: W0's frozen masks. Hollow: own masks, for comparison", color=INK, x=0.06, ha="left", fontsize=11)
    finish(fig, "agree_soft.png")


def rows() -> None:
    """Per-row means of the saved per-position forward KL on M at draw d1, float16 as stored."""
    files = {"W0_own": "a2_W0_M_d1", "A_own": "a2_A_M_d1", "N_own": "a2_N_M_d1", "Z": "a2_Z_M_d1", "B_own": "a1_B_M_d1", "L_own": "a1_L_M_d1", "A_frozen": "A_M_d1", "B_frozen": "B_M_d1"}
    out = {k: [float(x) for x in load_file(str(RES / "large" / f"{f}.safetensors"))["fwd"].astype("float64").mean(axis=1)] for k, f in files.items()}
    out["N_below_A_rows"] = sum(n < a for n, a in zip(out["N_own"], out["A_own"]))  # pyright: ignore[reportArgumentType]
    (RES / "rows.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    two_by_two()
    own_split()
    teeth()
    frozen()
    agree_soft()
    rows()
