"""E213 figure: the lowest R a circuit of k parts can have, against k, one line per trained fine-tune, one panel per
saved distance; E208's own circuits as points; E195's rebuilt circuit as the built-rewrite reference.

    ~/commute-agents/fleet/cpu-run python experiments/r_bounds_e208/plot.py
"""

import json
from pathlib import Path

import numpy as np

from commute import figures

HERE = Path(__file__).parent
S = json.loads((HERE / "results" / "summary.json").read_text())
P = json.loads((HERE / "results" / "predictions.json").read_text())["settings"]
TITLES = {"main": "the note: per-setting cos^2", "i": "the code's aggregate cos^2",
          "iii": "the note, M's settings only", "b": "size-aware distance"}
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def curve(item: dict) -> tuple[list[int], list[float]]:
    wk = item["worst_k"]
    ks = [int(k) for k in wk if k != "all"] + [0]
    ks[-1] = max(ks[:-1]) * 3  # 'all' drawn to the right of 4096
    return ks, [1 - v for v in wk.values()]


def main() -> None:
    fig, axs = figures.figure((7.5, 6.2), ncols=2, nrows=2)
    rows, items = [], S["items"]
    for ax, d in zip(axs.flat, P["distances"], strict=True):
        for c, (reuse, new) in zip(COLOURS, P["pairs"], strict=True):
            for name, style in ((reuse, "-"), (new, "--")):
                for seed in P["seeds"]:
                    it = items[f"{name}_s{seed}"][d]
                    ks, y = curve(it)
                    lab = f"{reuse}/{new}: {'reuse' if style == '-' else 'twin'}" if seed == 0 else None
                    ax.plot(ks, y, style, color=c, linewidth=0.9, label=lab)
                    ax.plot(np.mean(it["e208_K_size"]), 1 - it["e208_K_mean"], "o", color=c, markersize=3)
                    rows += [{"distance": d, "item": f"{name}_s{seed}", "k": k, "one_minus_R": v} for k, v in zip(ks, y, strict=True)]
        ks, y = curve(items["rebuilt"][d])
        ax.plot(ks, np.maximum(y, 1e-6), color="#555555", linewidth=1.4, label="built rewrite (E195 rebuilt)")
        ax.axhline(1 - P["level_bar"], color="#777777", linestyle=":", linewidth=0.8, label="R = 0.99")
        ax.axvline(P["k_min"], color="#bbbbbb", linestyle="-", linewidth=0.6)
        ax.set(xscale="log", yscale="log", ylim=(1e-5, 1.2), title=TITLES[d],
               xlabel="parts in the circuit, k (rightmost: all)", ylabel="1 - lowest possible R")
        ax.title.set_fontsize(8)
        ok = [k for k in [*map(str, P["k_grid"]), "all"]
              if all(items[f"{n}_s{s}"][d]["worst_k"][k] >= P["level_bar"] for pr in P["pairs"] for n in pr for s in P["seeds"])]
        print(f"{d}: every trained fine-tune has worst R >= {P['level_bar']} from k = {ok[0] if ok else 'never'}")
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="outside right center")
    fig.suptitle("How low can R go for a circuit of k parts? Dots: E208's own circuits", fontsize=10)
    figures.save(fig, HERE / "figures" / "worst_r.png", data=rows,
                 n="20 trained fine-tunes (5 pairs x seeds 0, 1; both seeds drawn); 12,312 to 12,314 parts each")


if __name__ == "__main__":
    main()
