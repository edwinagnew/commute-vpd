"""E143's figure, from results/stage1.json and the saved readings of E135 and E140 only (no model).

gaps.png: per scorecard pair with saved updates (rows) and reading (columns), the gap between the side built to reuse and the
side built to be new, each value a share of the fine-tune's own gain in log-odds of its target, on the rows the share was read
on. x = not read (no per-key vector on lowercase vs ciphers). Steering on copy vs lookup is E135's, on its 100 score rows; the
two label pairs are E140's, real minus shuffled twin, three seeds per side.

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 2G python experiments/holes_fill/plot.py [RESULTS FIGURES]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
COLS = {"share": "share", "steer_pooled": "steering,\npooled", "steer_perkey": "steering,\nper key", "neuron_k64": "neurons,\nk 64",
        "neuron_kmatched": "neurons,\nk matched"}  # fmt: skip
ROWS = {"e46_copy_random": "copy vs random maps", "e53_lowercase_cipher": "lowercase vs ciphers", "e54_matched": "half-known vs cipher, equal size",
        "e54_own": "half-known vs cipher, own step", "open_pair_c3": "rank-1 vs cipher", "open_pair_c3_copy": "copy vs cipher",
        "e62_copy_lookup": "copy vs lookup", "case": "case label, real vs twin", "repeat": "repeat label, real vs twin"}  # fmt: skip
E140_COLS = {"share": "share", "steer_pooled": "steer_pooled", "steer_perkey": "steer_perword", "neuron_k64": "neuron_k64", "neuron_kmatched": "neuron_kmatched"}


def e135_gaps() -> dict[str, float]:
    st = json.loads((EXP / "steering_copy_lookup" / "results" / "stage1.json").read_text())["states"]
    g = lambda kd, a: statistics.fmean(st[f"{a}_s{s}"]["steering"][kd]["share_lo"] for s in range(3))
    return {f"steer_{kd}": g(kd, "copy") - g(kd, "lookup") for kd in ("pooled", "perkey")}


def table(results: Path) -> tuple[list[list[float | None]], list[dict[str, Any]], dict[str, Any]]:
    score = json.loads((results / "stage1.json").read_text())["score"]
    e140 = json.loads((EXP / "real_vs_twin" / "results" / "stage3.json").read_text())["score"]["lo"]
    extra = {"e62_copy_lookup": e135_gaps()}
    values, recs = [], []
    for p in ROWS:
        row: list[float | None] = []
        for m in COLS:
            if p in ("case", "repeat"):
                v, src = e140[E140_COLS[m]][p]["gap"], "E140"
            elif m.startswith("steer") and p in extra:
                v, src = extra[p][m], "E135"
            else:
                rec = score["calls"][m].get(p)
                v, src = (rec["gap"] if rec else None), "E143"
            row.append(v)
            recs.append({"pair": p, "reading": m, "gap": v, "source": src})
        values.append(row)
    return values, recs, score


def main(results: Path = HERE / "results", figures: Path = HERE / "figures") -> None:
    values, recs, score = table(results)
    comp = score["compare"]
    fig, ax = FG.figure((7.5, 5.2))
    FG.gap_heatmap(ax, values, list(ROWS.values()), list(COLS.values()), label="gap, side built to reuse minus side built to be new,\nshare of the gain", vmax=1.0)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(f"Does steering or the neuron version separate the sides where the share does?\nOf the {comp['steering']['share_count']} "
                 f"pairs the share separates, steering separates {comp['steering']['count']} in its direction, neurons {comp['neuron']['count']}", fontsize=8)  # fmt: skip
    FG.save(fig, figures / "gaps.png", n="2 to 8 saved fine-tune seeds per side; read rows as the share's (75 to 500 per task); "
            "copy vs lookup steering and the two label pairs from earlier runs on the same rows; x = not read", data=recs)  # fmt: skip


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:3]))
