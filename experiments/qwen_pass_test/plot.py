"""E68's figures from the results JSONs alone; no model.

  figures/gap_closed.png    share of the KL gap closed per placement and update, with the faithful line
  figures/gap_closed_clean.png  the slides' four methods on the first update, on one reading (E68, E82)
  figures/gain_kept_clean.png  the same four, by the share of the drop in next-token loss kept under the masks
  figures/opening_kl.png    KL to the fine-tuned model against how far the masks are opened, update on and off
  figures/opening_nll.png   next-token NLL against the same opening, with clean W0 and the fine-tuned model

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/qwen_pass_test/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
INK, MUTED, GRID = "#2b2a27", "#6f6e69", "#e4e3df"
UPDATES = (("u1", "first fine-tune\nstep 10"), ("u2", "gold fine-tune\nstep 100"), ("u3", "template\nstep 100"),
           ("u4", "first fine-tune\nstep 100"))
ARMS = (("do_nothing", "do-nothing", "#8a8984", "o"), ("lower", "rarely-on half", "#2a78d6", "v"),
        ("random", "random change", "#eb6834", "X"), ("adjoint", "adjoint split", "#1baf7a", "s"),
        ("ls", "least-squares split", "#6250d6", "D"), ("upper", "often-on half", "#eda100", "^"),
        ("ceiling", "whole update, unmasked", "#e34948", "*"))
DO_NOTHING, CEILING = ARMS[0], ARMS[-1]
OPEN_ROWS, T = 32, (0.0, 0.1, 0.5, 1.0)


def load(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def gap_closed() -> None:
    sc, ceil = load("score.json"), load("amend1/score.json")
    fig, (top, low) = plt.subplots(2, 1, figsize=(9.5, 6.6), dpi=170, sharex=True, height_ratios=[1, 2])
    n = len(ARMS)
    for x, (u, _) in enumerate(UPDATES):
        k0, kd = sc[u]["K0"], sc[u]["KD"]
        line = 1 - 0.5 * k0 / (kd - k0)  # the faithful line K <= 1.5 K0 on this share's axis
        top.plot([x - 0.42, x + 0.42], [line, line], color=INK, lw=1.6, label="faithful line" if x == 0 else None)
        top.text(x, line + 0.03, f"{line:.2f}", ha="center", fontsize=8, color=INK)
        for i, (arm, label, col, mark) in enumerate(ARMS):
            v = ceil[u]["r2_C"] if arm == "ceiling" else sc[u]["states"][arm]["r2"]
            dx = (i - (n - 1) / 2) * 0.115
            size = 90 if mark == "*" else 38
            for ax in (top, low):
                ax.scatter(x + dx, v, color=col, marker=mark, s=size, edgecolor="white", linewidth=0.5, zorder=3,
                           label=label if x == 0 and ax is low else None)
            if arm in ("ls", "ceiling"):
                low.annotate(f"{v:.3f}", (x + dx, v), xytext=(0, 8), textcoords="offset points", ha="center",
                             fontsize=8, color=INK)
    top.set_ylim(-0.03, 1.08)
    low.set_ylim(-0.016, 0.16)
    top.set_ylabel("share of the KL gap closed\n(whole axis)")
    low.set_ylabel("share of the KL gap closed\n(zoom on the same numbers)")
    low.set_xticks(range(len(UPDATES)), [lab for _, lab in UPDATES])
    for ax in (top, low):
        style(ax)
        ax.axhline(0, color=MUTED, lw=0.8)
    top.set_title("No placement reaches the faithful line, and neither does the whole update unmasked", fontsize=11, color=INK, loc="left")
    handles = top.get_legend_handles_labels()[0] + low.get_legend_handles_labels()[0]
    labels = top.get_legend_handles_labels()[1] + low.get_legend_handles_labels()[1]
    low.legend(handles, labels, ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(FIG / "gap_closed.png")
    plt.close(fig)


def methods_on_first_update(state_key: str, teacher_key: str, whole: float, xlabel: str, title: str, name: str, line: float | None = None) -> None:
    """The slides' four methods on the first update, on one reading: E68's states, and E82's read of E76's clean-loss state. `whole` is a reference, not a method."""
    st, teacher = load("score.json")["u1"]["states"], load("teacher_clean.json")
    rows = (("Do-nothing", st["do_nothing"][state_key], "#8a8984", "s"), ("Least squares", st["ls"][state_key], "#7a4cc9", "D"),
            ("Adjoint split", st["adjoint"][state_key], "#e07a00", "^"), ("Teacher training", teacher[teacher_key], "#2a78d6", "o"))
    fig, ax = plt.subplots(figsize=(9.5, 3.9), dpi=170)
    box = {"fc": "white", "ec": "none", "pad": 1}
    for y, (label, v, col, mark) in enumerate(rows):
        ax.plot([0, v], [y, y], color=col, lw=2.2, zorder=2)
        ax.scatter(v, y, marker=mark, s=120, color=col, linewidth=2, zorder=3)
        ax.text(v + 0.018, y, f"{v:.2f}", va="center", fontsize=10, color=INK, zorder=4, bbox=box)
    ax.axvline(whole, color=MUTED, lw=1.4, ls="--", zorder=1)
    ax.text(whole + 0.012, -0.62, "whole update added directly", ha="left", va="center", fontsize=10, color=MUTED, bbox=box)
    if line is not None:
        ax.axvline(line, color=INK, lw=1.6, zorder=1)
        ax.text(line - 0.012, -0.62, "faithful line", ha="right", va="center", fontsize=10, color=INK, bbox=box)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=12)
    ax.set_ylim(len(rows) - 0.5, -0.85)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, color=INK, loc="left", x=-0.17)
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def gap_closed_clean() -> None:
    a1, line = load("amend1/score.json")["u1"], load("teacher_clean.json")["faithful_line"]
    methods_on_first_update("r2", "share_closed", a1["r2_C"], "Share of the KL gap closed",
                            "On Qwen, no method carries the update through the masks,\nand neither does adding the whole update", "gap_closed_clean.png", line)
    methods_on_first_update("gain_kept", "gain_kept", a1["gain_C"], "Share of the fine-tune's drop in next-token loss kept under the masks",
                            "Under the masks, teacher training keeps a quarter of the gain,\nabout what adding the whole update keeps", "gain_kept_clean.png")


def row_mean(rd: dict, key: str, per_row: int) -> float:
    xs = rd["sums"][key][:OPEN_ROWS]
    return sum(xs) / (len(xs) * per_row)


def opening_read(u: str, on: bool, t: float) -> dict:
    if t == 0:
        return load(f"amend1/{u}_{'ceiling' if on else 'zero'}.json")
    return load(f"amend1/{u}_open_t{t}_{'on' if on else 'off'}.json")


def opening(key: str, per: str, ylabel: str, title: str, refs: dict, ylim: tuple[float, float] | None, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), dpi=170, sharey=True)
    for ax, (u, lab) in zip(axes, UPDATES[:2], strict=True):
        rd0 = opening_read(u, True, 0.0)
        per_row = rd0["T"] if per == "T" else rd0["T"] - 1
        for on, arm, label in ((True, CEILING, "whole update added"), (False, DO_NOTHING, "nothing added")):
            ys = [row_mean(opening_read(u, on, t), key, per_row) for t in T]
            ax.plot(T, ys, color=arm[2], marker=arm[3], ms=10 if arm[3] == "*" else 6, lw=2, label=label, zorder=3)
            for t, y in zip(T, ys, strict=True):
                if t in (0.5, 1.0):
                    low = y < 0.1  # under the faithful line's label: beside the point, not above it
                    ax.annotate(f"{y:.3f}" if key == "nll" or low else f"{y:.2f}", (t, y), xytext=(-9, 0) if low else (-4, 7),
                                textcoords="offset points", ha="right", va="center" if low else "baseline", fontsize=8, color=INK)
        for label, value, ls in refs[u]:
            ax.axhline(value, color=MUTED if ls != "-" else INK, lw=1.1, ls=ls, zorder=1)
            ax.text(0.5, value, f"{label} {value:.3f}", fontsize=7.5, color=INK, ha="center", va="bottom",
                    bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.85})
        ax.set_title(lab.replace("\n", ", "), fontsize=10, color=INK)
        ax.set_xlabel("how far the masks are opened: t in g + t (1 − g)")
        ax.set_xticks(T)
        style(ax)
    axes[0].set_ylabel(ylabel)
    if ylim:
        axes[0].set_ylim(*ylim)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, fontsize=8, frameon=False, loc="lower center")
    fig.suptitle(title + f" (first {OPEN_ROWS} rows)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / name)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    gap_closed()
    gap_closed_clean()
    op, w0 = load("amend1/open.json"), load("u1/W0.json")
    line = 1.5 * row_mean(w0, "ci", w0["T"])
    kl_refs = {u: (("clean W0", op[u]["clean"], "--"), ("faithful line", line, ":")) for u, _ in UPDATES[:2]}
    opening("K", "T", "KL(fine-tuned ‖ this forward)", "On KL, the whole update rebuilds the fine-tuned model only with the masks fully open", kl_refs, None,
            "opening_kl.png")
    nll_refs = {u: (("clean W0", row_mean(load(f"{u}/W0.json"), "clean", w0["T"] - 1), "--"),
                    ("fine-tuned model", row_mean(load(f"{u}/W0.json"), "ft", w0["T"] - 1), "-")) for u, _ in UPDATES[:2]}
    opening("nll", "T1", "next-token NLL, nats", "Half-open masks lower the NLL below clean W0 with no update at all", nll_refs, None,
            "opening_nll.png")


if __name__ == "__main__":
    main()
