"""E58 Amendment 3's figures, one per claim, from results/amend3.json (job 46398), with the 24-key references from
results/run.json (E58, job 46266) and results/amend1.json (job 46332). No model.

  figures/a3_carry.png  left: the elicit map's held-out carry x at every kept checkpoint, 128 keys against E58's 24-key
                        state re-trained in the same job; the cipher control at steps 10-16 and 30; the shuffled elicit
                        map at step 10; the control's +-0.05 band. Right: the same elicit states' trained-key and
                        held-out value-specific gains in log-odds, the two halves of x
  figures/a3_shift.png  r and the stand-in ratio against x, each map moved from 24 keys (three seeds, mean) to 128 keys
                        (step 10 scored, step 30 companion); the cipher's move is the drift line, with the filed
                        +-0.05 guard around its 24-key mean

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/open_pair/plot_amend3.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import COLOR, FIG, GRID, HERE, INK, SHADE, style

STEPS = (10, 30, 100, 300)
MAPS = (
    ("elicit", "elicit (half-known)"),
    ("rank1", "rank-1 (W0's top word)"),
    ("copy", "copy (value = key)"),
    ("cipher", "cipher (no rule)"),
)


def load() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    a3 = json.loads((HERE / "results" / "amend3.json").read_text())
    run = json.loads((HERE / "results" / "run.json").read_text())["states"]
    a1 = json.loads((HERE / "results" / "amend1.json").read_text())["control"]
    ref = {m: [run[f"{m}_s{s}"] for s in range(3)] for m in ("elicit", "cipher")}
    ref.update({m: [a1[f"{m}_s{s}"] for s in range(3)] for m in ("rank1", "copy")})
    return a3, ref


def carry_fig(a3: dict[str, Any]) -> None:
    rd = a3["reads"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.6), sharex=True)
    ax.axhspan(-0.05, 0.05, color=SHADE, zorder=0)
    ax.text(
        10.3,
        0.055,
        "control passes if x within ±0.05; below it reads 'below' (no carry)",
        fontsize=6.5,
        color=INK,
    )
    ax.axhline(0, color=INK, lw=0.7)
    ax.plot(
        STEPS,
        [rd[f"elicit@{s}"]["x"] for s in STEPS],
        "-o",
        color=COLOR["elicit"],
        label="elicit, 96 trained keys",
    )
    ax.plot(
        STEPS,
        [rd[f"elicit24@{s}"]["x"] for s in STEPS],
        "--o",
        color=COLOR["elicit"],
        mfc="white",
        label="elicit, 24 trained keys (E58's state, re-trained)",
    )
    cs = [10, 11, 12, 13, 14, 15, 16]
    ax.plot(cs, [rd[f"cipher@{s}"]["x"] for s in cs], "-", color=COLOR["cipher"], lw=1)
    ax.scatter(
        cs[:-1],
        [rd[f"cipher@{s}"]["x"] for s in cs[:-1]],
        s=18,
        color="white",
        edgecolors=COLOR["cipher"],
        zorder=3,
        label="cipher, 96 keys, under the gate (unreadable)",
    )
    ax.scatter(
        [16, 30],
        [rd["cipher@16"]["x"], rd["cipher@30"]["x"]],
        s=36,
        color=COLOR["cipher"],
        zorder=3,
        label="cipher, 96 keys: control read at step 16; step 30",
    )
    sh = rd["shuffle@10"]
    ax.scatter(
        [10],
        [sh["x"]],
        marker="x",
        s=40,
        color="#6b6a66",
        zorder=3,
        label="shuffled elicit, step 10 (under the gate: unreadable)",
    )
    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(s) for s in STEPS])
    ax.set_xlabel("fine-tune step", color=INK)
    ax.set_ylim(-1.05, 0.35)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    style(
        ax,
        "Held-out carry x: 0 = unseen keys gained nothing, 1 = as much as trained keys",
        "held-out carry x (value-specific)",
    )
    for tag, ls, mfc, keys in (
        ("elicit", "-", None, "96"),
        ("elicit24", "--", "white", "24"),
    ):
        g = [rd[f"{tag}@{s}"]["G_spec"] for s in STEPS]
        h = [rd[f"{tag}@{s}"]["H_spec"] for s in STEPS]
        bx.plot(
            STEPS,
            g,
            ls,
            marker="s",
            color="#2a2a2a",
            mfc=mfc or "#2a2a2a",
            label=f"trained keys, {keys} keys",
        )
        bx.plot(
            STEPS,
            h,
            ls,
            marker="o",
            color=COLOR["elicit"],
            mfc=mfc or COLOR["elicit"],
            label=f"held-out keys, {keys} keys",
        )
    bx.axhline(0, color=INK, lw=0.7)
    bx.text(10.3, 0.4, "W0 / zero update: 0", fontsize=6.5, color=INK)
    bx.set_xlabel("fine-tune step", color=INK)
    bx.legend(fontsize=6.5, loc="center right", frameon=False)
    style(
        bx,
        "The two halves of the elicit map's x",
        "value-specific gain, log-odds\n(fine-tuned minus W0)",
    )
    fig.suptitle(
        "With 96 trained keys, does the half-known map carry to unseen keys, and does the carry last?",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "a3_carry.png", dpi=150)
    plt.close(fig)


def at_128(a3: dict[str, Any], m: str, step: int) -> dict[str, Any]:
    return a3["reads"][
        f"{m}@{a3['cipher_matched_step'] if m == 'cipher' and step == 10 else step}"
    ]


def shift_fig(a3: dict[str, Any], ref: dict[str, list[dict[str, Any]]]) -> None:
    th = json.loads((HERE / "results" / "predictions_amend3.json").read_text())[
        "thresholds"
    ]
    cip24 = sum(r["r"] for r in ref["cipher"]) / 3
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True)
    for ax, (q, name, tol) in zip(
        axes,
        (
            ("r", "re-gating share r", th["tol_r"]),
            ("ratio", "stand-in ratio (no ceiling at 1)", th["tol_ratio"]),
        ),
        strict=True,
    ):
        if q == "r":
            ax.axhspan(cip24 - tol, cip24 + tol, color=SHADE, zorder=0)
            ax.text(
                -0.12,
                cip24 - tol + 0.01,
                "drift guard: the cipher's r must stay within ±0.05 of its 24-key mean",
                fontsize=6.5,
                color=INK,
            )
        for m, label in MAPS:
            xs0, ys0 = [r["x"] for r in ref[m]], [r[q] for r in ref[m]]
            x0, y0 = sum(xs0) / 3, sum(ys0) / 3
            p10, p30 = at_128(a3, m, 10), at_128(a3, m, 30)
            if m != "cipher":
                ax.fill_between(
                    [x0 - 0.03, p10["x"] + 0.03],
                    y0 - tol,
                    y0 + tol,
                    color=COLOR[m],
                    alpha=0.10,
                    lw=0,
                )
            ax.scatter(xs0, ys0, s=14, color=COLOR[m], alpha=0.45, zorder=3)
            ax.annotate(
                "",
                (p10["x"], p10[q]),
                (x0, y0),
                arrowprops={"arrowstyle": "->", "color": COLOR[m], "lw": 1.3},
            )
            ax.scatter(
                [p10["x"]],
                [p10[q]],
                s=44,
                color=COLOR[m],
                zorder=4,
                label=f"{label}: 24 → 96 keys, step 10",
            )
            ax.scatter(
                [p30["x"]],
                [p30[q]],
                s=40,
                marker="D",
                color="white",
                edgecolors=COLOR[m],
                zorder=4,
            )
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlabel("held-out carry x (value-specific)", color=INK)
        style(ax, f"shaded: each map's 24-key value ± {tol}, the filed tolerance", name)
    axes[1].scatter(
        [],
        [],
        marker="D",
        color="white",
        edgecolors=INK,
        label="96 keys, step 30 (companion, not scored)",
    )
    axes[1].scatter(
        [],
        [],
        s=14,
        color=INK,
        alpha=0.45,
        label="24 keys, three seeds (arrow starts at their mean)",
    )
    axes[0].set_xlim(-0.55, 0.95)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "From 24 to 96 trained keys, carry rose on every map: did r and the stand-in ratio move with it?",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "a3_shift.png", dpi=150)
    plt.close(fig)


def main() -> None:
    a3, ref = load()
    FIG.mkdir(exist_ok=True)
    carry_fig(a3)
    shift_fig(a3, ref)
    for m, _ in MAPS:
        p = at_128(a3, m, 10)
        print(
            m,
            "24 keys x/r/ratio",
            [round(sum(r[k] for r in ref[m]) / 3, 3) for k in ("x", "r", "ratio")],
            "| 96 keys step",
            p["step"],
            [round(p[k], 3) for k in ("x", "r", "ratio")],
        )


if __name__ == "__main__":
    main()
