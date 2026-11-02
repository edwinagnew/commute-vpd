"""The re-gating synthesis figure, from the E58 results only (no model):

  figures/synthesis.png  held-out carry x against r (left) and the stand-in ratio (right) for every E58 state with a
                         readable label: the elicit, rank-1, copy and cipher maps at 24 trained keys (run.json,
                         amend1.json; step 10, the cipher at its update-size-matched step 8-9) and the elicit, rank-1,
                         copy, cipher and shuffled maps at 96 trained keys (amend3.json seed 0, amend4.json seeds 1-2 and
                         the ciphers, amend5.json shuffle seeds 1-2; step 10 or the first step over the gate, and step
                         30). Each point is a three-seed mean with its seed range drawn on both axes. E54's and E56's
                         states train all 32 keys, so they have no held-out x: their r and ratio ranges are drawn as
                         horizontal bands (E56's five seeds for r, E54's two seeds for the ratio).

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/open_pair/plot_synthesis.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import COLOR, FIG, GRID, HERE, INK, style

COLOR = {**COLOR, "rank1": "#7b5ab8", "copy": "#2a78d6", "shuffle": "#6b6a66"}
NAME = {
    "elicit": "elicit (half-known)",
    "rank1": "rank-1 (W0's top word)",
    "copy": "copy (value = key)",
    "cipher": "cipher (no rule)",
    "shuffle": "shuffled elicit",
}
RES = HERE / "results"


def js(name: str) -> dict[str, Any]:
    return json.loads((RES / name).read_text())


def groups() -> list[tuple[str, int, str, list[dict[str, Any]]]]:
    """(map, trained keys, stage, three seed reads). stage: 'first' = step 10, or the first step over the gate for the
    96-key cipher and shuffle, or the update-size-matched step for the 24-key cipher; 'step 30'."""
    run, a1, a3, a4, a5 = (
        js("run.json")["states"],
        js("amend1.json")["control"],
        js("amend3.json")["reads"],
        js("amend4.json"),
        js("amend5.json"),
    )
    r4 = a4["reads"]
    out: list[tuple[str, int, str, list[dict[str, Any]]]] = [
        ("elicit", 24, "first", [run[f"elicit_s{s}"] for s in range(3)]),
        ("cipher", 24, "first", [run[f"cipher_s{s}"] for s in range(3)]),
        ("rank1", 24, "first", [a1[f"rank1_s{s}"] for s in range(3)]),
        ("copy", 24, "first", [a1[f"copy_s{s}"] for s in range(3)]),
    ]
    for m in ("elicit", "rank1", "copy"):
        for st in (10, 30):
            out.append(
                (
                    m,
                    96,
                    "first" if st == 10 else "step 30",
                    [a3[f"{m}@{st}"], r4[f"{m}_s1@{st}"], r4[f"{m}_s2@{st}"]],
                )
            )
    out.append(
        (
            "cipher",
            96,
            "first",
            [r4[f"cipher_s{s}@{a4['control'][str(s)]['step']}"] for s in range(3)],
        )
    )
    out.append(("cipher", 96, "step 30", [r4[f"cipher_s{s}@30"] for s in range(3)]))
    sh1 = a5["reads"]
    out.append(
        (
            "shuffle",
            96,
            "first",
            [
                r4[f"shuffle_s0@{a4['shuffle_gate_step']}"],
                sh1[f"shuffle_s1@{a5['gate_step_s1']}"],
                sh1[f"shuffle_s2@{a5['gate_step_s2']}"],
            ],
        )
    )
    out.append(
        (
            "shuffle",
            96,
            "step 30",
            [r4["shuffle_s0@30"], sh1["shuffle_s1@30"], sh1["shuffle_s2@30"]],
        )
    )
    return out


def bands() -> dict[str, dict[str, tuple[float, float]]]:
    e56 = js("../../keydependent_patch/results/job2.json")["P2"]["r"]
    table = json.loads(
        (HERE.parent / "regating_truth_table" / "results" / "table.json").read_text()
    )["runs"]
    ratio = {
        m: [table[f"e54/{m}_s{s}"]["ratio"]["value"] for s in (0, 1)]
        for m in ("elicit", "cipher")
    }
    return {
        "r": {
            "elicit": (min(e56["a"]), max(e56["a"])),
            "cipher": (min(e56["b"]), max(e56["b"])),
        },
        "ratio": {m: (min(v), max(v)) for m, v in ratio.items()},
    }


def main() -> None:
    gs, bd = groups(), bands()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharex=True)
    for ax, q, name in (
        (axes[0], "r", "r: share of the update's gain\non W0's own subcomponents"),
        (axes[1], "ratio", "stand-in ratio (no ceiling at 1)"),
    ):
        source = "E56, 5 seeds" if q == "r" else "E54, 2 seeds"
        for m, (lo, hi) in bd[q].items():
            ax.axhspan(lo, hi, color=COLOR[m], alpha=0.12, lw=0, zorder=0)
            ax.text(
                0.93,
                hi + 0.008,
                f"{source}: {m}, all 32 keys trained (no x)",
                fontsize=6.5,
                color=INK,
                ha="right",
            )
        ax.axhline(0.5, color=INK, lw=1.0, ls=":")
        ax.text(
            0.93,
            0.505,
            "0.5: the reading's cut",
            fontsize=6.5,
            color=INK,
            ha="right",
        )
        for m, keys, stage, reads in gs:
            xs, ys = [d["x"] for d in reads], [d[q] for d in reads]
            mx, my = sum(xs) / 3, sum(ys) / 3
            ax.plot([min(xs), max(xs)], [my, my], color=COLOR[m], lw=1.0, zorder=2)
            ax.plot([mx, mx], [min(ys), max(ys)], color=COLOR[m], lw=1.0, zorder=2)
            ax.scatter(
                [mx],
                [my],
                s=58 if keys == 96 else 46,
                marker="o" if keys == 96 else "^",
                zorder=3,
                color=COLOR[m] if stage == "first" else "white",
                edgecolors=COLOR[m],
                lw=1.4,
            )
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlabel(
            "held-out carry x (value-specific): 0 = unseen keys gained nothing",
            color=INK,
        )
        style(ax, "three-seed means; lines = seed ranges", name)
    axes[0].set_ylim(0.15, 0.97)
    axes[1].set_ylim(0.2, 1.28)
    axes[0].set_xlim(-0.45, 0.95)
    for m in ("elicit", "rank1", "copy", "cipher", "shuffle"):
        axes[1].scatter([], [], s=40, color=COLOR[m], label=NAME[m])
    axes[1].scatter(
        [],
        [],
        s=40,
        marker="^",
        color=INK,
        label="24 trained keys, step 10 (cipher: step 8-9)",
    )
    axes[1].scatter(
        [],
        [],
        s=40,
        marker="o",
        color=INK,
        label="96 trained keys, first read (filled):\nstep 10; cipher, shuffle: step 15-16",
    )
    axes[1].scatter(
        [],
        [],
        s=40,
        marker="o",
        color="white",
        edgecolors=INK,
        label="96 trained keys, step 30 (hollow)",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Does r separate maps by whether the target is the model's own guess, or by how far the rule carries?",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "synthesis.png", dpi=150)
    plt.close(fig)
    for m, keys, stage, reads in gs:
        print(
            m,
            keys,
            stage,
            [round(d["x"], 3) for d in reads],
            [round(d["r"], 3) for d in reads],
            [round(d["ratio"], 3) for d in reads],
        )


if __name__ == "__main__":
    main()
