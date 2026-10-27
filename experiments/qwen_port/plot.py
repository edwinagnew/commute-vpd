"""Where the torch port's lower count of active subcomponents comes from (analysis
reports/analyses/2026-10-27_analysis_qwen-port.md, "Why torch counts fewer active subcomponents").

Reads JAX's lower importances g on the two detail rows (results/large/gamma_row{0,4}.safetensors, bf16) and counts,
for each threshold t, the (position, subcomponent) pairs with g > t, averaged over positions and summed over the 196
sites: the l0 count that jax_reference.py and torch_check.py report at t = 0. Torch's count at t = 0 comes from
results/torch_check.json (torch's g itself was not saved). Writes results/l0_threshold.json (the curves, plus the
share of the summed g carried by values below 1e-3) and figures/l0_threshold.png.

Usage (repo root, a CPU job with ~8 GB): uv run python experiments/qwen_port/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
RESULTS, LARGE, FIGURES = HERE / "results", HERE / "results" / "large", HERE / "figures"
ROWS = {0: "text row 0", 4: "GSM8K prompt, row 4"}
COLOR = {0: "#2a78d6", 4: "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
THRESHOLDS = [0.0, *np.logspace(-7, -0.5, 66).tolist()]
SMALL = 1e-3
# one bf16 rounding step for terms between 1/16 and 1/8: the largest group of leftovers
BF16_STEP = 2.0**-11


def curve(row: int) -> dict:
    f = load_file(LARGE / f"gamma_row{row}.safetensors")
    gs = [f[k].float() for k in f if k.startswith("g.")]
    n_pos = gs[0].shape[0]
    counts = [sum(float((g > t).sum()) for g in gs) / n_pos for t in THRESHOLDS]
    total = sum(float(g.sum()) for g in gs) / n_pos
    small = sum(float(g[(g > 0) & (g <= SMALL)].sum()) for g in gs) / n_pos
    return {
        "thresholds": THRESHOLDS,
        "jax_count": counts,
        "sum_g_per_position": total,
        "sum_g_below_1e-3_per_position": small,
    }


def main() -> None:
    torch_l0 = [
        r["l0_total"]
        for r in json.loads((RESULTS / "torch_check.json").read_text())["check2"][
            "all_rows"
        ]
    ]
    out = {str(r): {**curve(r), "torch_count_at_0": torch_l0[r]} for r in ROWS}
    (RESULTS / "l0_threshold.json").write_text(json.dumps(out, indent=1))

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r, name in ROWS.items():
        c = out[str(r)]
        ax.plot(
            c["thresholds"][1:],
            c["jax_count"][1:],
            color=COLOR[r],
            lw=2,
            label=f"JAX, {name}",
        )
        ax.axhline(
            c["torch_count_at_0"],
            color=COLOR[r],
            lw=1.5,
            ls="--",
            label=f"torch's count of g > 0, {name}",
        )
    ax.axvline(BF16_STEP, color=MUTED, lw=1, ls=":")
    ax.text(
        BF16_STEP * 1.15,
        ax.get_ylim()[1] * 0.995,
        "one bf16 step for\nterms near 0.1",
        color=MUTED,
        va="top",
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("threshold t on the lower importance g")
    ax.set_ylabel("subcomponents with g > t, per position")
    ax.set_title("JAX's extra active subcomponents all have g below 0.001", color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "l0_threshold.png", dpi=150)
    for r in ROWS:
        c = out[str(r)]
        at = dict(zip(c["thresholds"], c["jax_count"], strict=True))
        print(
            r,
            "jax at 0:",
            at[0.0],
            "torch at 0:",
            c["torch_count_at_0"],
            "sum g:",
            c["sum_g_per_position"],
            "sum g <= 1e-3:",
            c["sum_g_below_1e-3_per_position"],
        )
    print("figure:", FIGURES / "l0_threshold.png")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
