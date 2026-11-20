"""E229's figure from results/summary.json (no model): for each counted pair and seed, the real-minus-twin gap in R on
the read positions, upper tail only (this run) beside E217's two-sided reading at the same percentile.

  figures/gaps_upper_vs_two_sided.png

Usage (repo root):  ~/commute-agents/fleet/cpu-run .venv/bin/python experiments/topk_upper/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as fg

HERE = Path(__file__).resolve().parent
PAIRS = {"copy24/cipher24": "24-key copy vs cipher", "ident32/rand32": "identity map vs random map",
         "case_real/case_twin": "case task vs twin", "repeat_real/repeat_twin": "repeat task vs twin"}  # fmt: skip
BANDS = {"p99": "p99/1", "p97": "p97/3", "p90": "p90/10"}  # this run's band: E217's band at the same percentile


def main() -> None:
    gaps = {(g["band"], g["pair"]): g for g in json.loads((HERE / "results" / "summary.json").read_text())["gaps"]}
    assert len(gaps) == len(PAIRS) * len(BANDS), len(gaps)
    rows, labels, values = [], [], []
    for pair, name in PAIRS.items():
        for seed in ("0", "1"):
            labels.append(f"{name}, seed {seed}")
            line = []
            for band, e217 in BANDS.items():
                g = gaps[(band, pair)]
                up, two = g["gap_by_seed"][seed], g["e217_gap_by_seed"][seed]
                line += [up, two]
                rows.append({"pair": pair, "seed": seed, "band": band, "upper_gap": up, "e217_band": e217, "e217_gap": two})
            values.append(line)
    cols = [c for band, e217 in BANDS.items() for c in (f"{band}\nupper", f"{e217}\nE217")]
    fig, ax = fg.figure((7.5, 4.6))
    fg.gap_heatmap(ax, values, labels, cols, label="R, side built to reuse minus its twin", fmt=".4f")
    ax.set_title("With the upper tail only, the real-minus-twin gaps in R are several times smaller than E217's,\n"
                 "and at p99 two pairs reverse sign on both seeds", fontsize=9)
    fg.save(fig, HERE / "figures" / "gaps_upper_vs_two_sided.png", n="one fine-tune per side per seed, 512 windows each; "
            "E217 columns: its two-sided reading of the same fine-tunes (above p_hi or below p_lo)", data=rows)  # fmt: skip


if __name__ == "__main__":
    main()
