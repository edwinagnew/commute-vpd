"""E184's one figure (cell P1.20), from results/<item>.json and results/predictions.json only (no model).

Left: for each item with a filed expected rung, the rung read on list "all" at each bar minus the filed rung (0 when the
read rung is one of the filed rungs; the filed rung is stated for bar 0.9, the other bars are shown beside it). Right:
for each old pair and seed, the reuse side's rung minus the new side's at bar 0.9 (at or below 0 holds).

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/simplest_intervention/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from commute import figures as FG

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
BARS = ("0.8", "0.9", "0.95")
ITEMS = {  # run-1 controls, then run 2's items with a filed rung
    "do_nothing": "do-nothing", "l1_neuron": "one neuron shifted", "l2_triggered": "neuron shifted by a direction",
    "l4_bit": "random bit per window", "planted_constant": "planted constant", "planted_trigger": "planted trigger",
    "cue": "cue", "floor": "random bit per pair", "xor": "XOR", "twin_s0": "twin, seed 0", "twin_s1": "twin, seed 1",
    "fact_s0": "fact, seed 0", "fact_s1": "fact, seed 1",
}
PAIRS = {("copy24", "cipher24"): "copy vs cipher", ("ident32", "rand32"): "identity vs random",
         ("case_real", "case_twin"): "case vs its twin", ("repeat_real", "repeat_twin"): "repeat vs its twin"}


def rung(name: str, bar: str) -> int:
    return json.loads((RES / f"{name}.json").read_text())["readings"]["all"][bar]["rung"]


def main() -> None:
    expected = json.loads((RES / "predictions.json").read_text())["expected"]
    rows, left, data = [], [], []
    for item, label in ITEMS.items():
        filed = expected[item]["rung"]
        got = [rung(item, b) for b in BARS]
        left.append([0 if g in filed else g - min(filed) for g in got])
        rows.append(f"{label} (filed {'/'.join(map(str, filed))})")
        data += [{"panel": "built", "item": item, "bar": b, "filed": min(filed), "read": g, "diff": d}
                 for b, g, d in zip(BARS, got, left[-1], strict=True)]
    right: list[list[int]] = []
    for (reuse, new), label in PAIRS.items():
        right.append([rung(f"{reuse}_s{s}", "0.9") - rung(f"{new}_s{s}", "0.9") for s in (0, 1)])
        data += [{"panel": "pairs", "item": label, "bar": "0.9", "seed": s, "diff": d} for s, d in enumerate(right[-1])]
    fig, (a, b) = FG.figure(FG.FULL_WIDTH, ncols=2)
    FG.gap_heatmap(a, left, rows, [f"bar {x}" for x in BARS], label="read rung minus filed rung", vmax=3, fmt=".0f",
                   colorbar=False)
    a.set_title("Controls as built; 7 of 9 other items above their filed rung")
    FG.gap_heatmap(b, right, list(PAIRS.values()), ["seed 0", "seed 1"], vmax=3, fmt=".0f",
                   label="rung difference (left: read minus filed; right: reuse side minus new side)")
    b.set_title("Old pairs at bar 0.9: reuse side minus new side")
    fig.suptitle("E184: the ranking reads its controls as built, its other built items too high")
    FG.save(fig, HERE / "figures" / "rungs.png", n="one fine-tune per row and seed; 16,384 read windows per reading; "
            "rungs: 0 none, 1 re-setting, 2 new trigger, 3 new direction, 4 from scratch", data=data)


if __name__ == "__main__":
    main()
