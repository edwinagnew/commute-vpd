"""E235 run 1's figure, from saved results only (CPU): the share of each side's activation change carried by
fine-tune-specific latents, for the grafted heads (the planted control), the matched Gaussian and the primary pair's
two sides, by block and seed, with the planted bar drawn.

    ~/commute-agents/fleet/cpu-run python3 experiments/crosscoder_proper/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute.figures import figure, save

HERE = Path(__file__).resolve().parent
ITEMS = HERE / "results/run1/items"
BAR = json.loads((HERE / "results/predictions.json").read_text())["thresholds"]["full"]["planted_part"]
SIDE = {"reuse": [f"e54__elicit_s{s}.B.c0" for s in (0, 1, 2)] + [f"e54__elicit_s0.B.c{c}" for c in (1, 2)],
        "new": [f"e54__cipher_s{s}.B.c0" for s in (0, 1, 2)] + [f"e54__cipher_s0.B.c{c}" for c in (1, 2)]}  # fmt: skip
GROUPS = {  # column label: (item keys, colour)
    "block 2\ngrafted head": ([f"head.A.c{c}" for c in (0, 1, 2)], "#009E73"),
    "block 2\nhead,\nk 32 and 200": (["head.A.c0.k32", "head.A.c0.k200"], "#009E73"),
    "block 2\nGaussian\nand pair": ([], "#555555"),
    "block 3\ngrafted head": ([f"head.B.c{c}" for c in (0, 1, 2)], "#009E73"),
    "block 3\nmatched\nGaussian": ([f"gauss.B.c{c}" for c in (0, 1, 2)], "#555555"),
    "block 3\nelicit,\nbuilt reuse": (SIDE["reuse"], "#0072B2"),
    "block 3\ncipher,\nbuilt new": (SIDE["new"], "#D55E00"),
}


def specific(key: str) -> float:
    return float(json.loads((ITEMS / f"{key}.json").read_text())["read"]["specific"])


def main() -> None:
    fig, ax = figure((7.5, 3.8))
    rows = []
    for i, (label, (keys, colour)) in enumerate(GROUPS.items()):
        values = [specific(k) for k in keys]
        rows += [{"column": label.replace("\n", " "), "item": k, "specific": v} for k, v in zip(keys, values, strict=True)]
        ax.scatter([i] * len(values), values, color=colour, s=26, alpha=0.85, linewidths=0)
        if not keys:
            ax.text(i, 0.03, "not run:\nblock 2's\nplanted\ncontrol\nfailed", ha="center", va="center", fontsize=7, color="#555555")
    gauss = [specific(k) for k in GROUPS["block 3\nmatched\nGaussian"][0]]
    ax.fill_between([4.6, 6.4], min(gauss), max(gauss), color="#555555", alpha=0.2, linewidth=0)
    ax.text(6.45, max(gauss), "Gaussian's\nrange", fontsize=7, color="#555555", va="center")
    ax.axhline(BAR, color="#777777", linewidth=0.8, linestyle="--")
    ax.text(-0.4, BAR, f"planted bar, {BAR}", fontsize=7, color="#555555", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlim(-0.5, 7.2)
    ax.set_xticks(range(len(GROUPS)), list(GROUPS), fontsize=7)
    ax.set_ylabel("share of the change carried\nby fine-tune-specific latents")
    ax.set_title("Only block 3's grafted head clears the bar; there the Gaussian\nreads like both sides of the pair", fontsize=9)
    save(fig, HERE / "figures/planted_gauss_pair.png",
         n="Each point is one crosscoder. Heads and Gaussian: 3 crosscoder seeds. Pair sides: 3 fine-tune seeds at "
           "crosscoder seed 0, plus 2 crosscoder seeds on fine-tune seed 0. k sweep: block 2 head, seed 0.",
         data=rows)  # fmt: skip


if __name__ == "__main__":
    main()
