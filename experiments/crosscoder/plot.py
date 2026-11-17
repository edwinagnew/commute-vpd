"""E198's scorecard columns, from saved results only (CPU, after run.py): the crosscoder reading d (the change's shared
part minus its fine-tune-specific part) per built pair, as built and size-matched, beside the re-gating share on the
part 2 scorecard's 67M panel (experiments/method_table/gate_plot_1116.py), same colour scale, same seed-range rule
(`plot_final.score`). Descriptive: no reuse/new verdict. Rules: results/predictions.json "rules".

Writes results/summary.json and figures/scorecard_crosscoder.png (+ .pdf, .csv).

    ~/commute-agents/fleet/cpu-run --mem 6G python experiments/crosscoder/plot.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from commute.figures import figure, save
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
RES, REPO = HERE / "results", HERE.parents[1]
GP = load_experiment("method_table", "gate_plot_1116")  # cells_67m, draw, records, COLS: the scorecard's 67M panel
PF = load_experiment("method_table", "plot_final")  # score: the scorecard's rule
SM = load_experiment("size_matched", "run")  # pairs(): the share's states per pair
SMATCH = REPO / "experiments/size_matched/results/summary.json"
NEW_COLS = {"crosscoder": "crosscoder:\nshared minus\nfine-tune-only", "share_matched": "re-gating\nshare,\nsize-matched",
            "crosscoder_matched": "crosscoder,\nsize-matched"}  # fmt: skip


def key(state: str) -> str:
    return state.replace("/", "__")


def cells(sides: dict[str, dict[str, Any]], m67: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per built pair, the crosscoder cell as built and size-matched (larger side scaled, smaller side as built)."""
    names, _, _ = SM.pairs()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, (r, n) in names.items():
        larger = m67[pair]["larger"]
        for col, kr, kn in (("crosscoder", [key(s) for s in r], [key(s) for s in n]),
                            ("crosscoder_matched", [key(s) + (f"__matched_{pair}" if larger == "reuse" else "") for s in r],
                             [key(s) + (f"__matched_{pair}" if larger == "new" else "") for s in n])):  # fmt: skip
            vals = {"reuse": [sides[k]["d"] for k in kr if k in sides], "new": [sides[k]["d"] for k in kn if k in sides]}
            if len(vals["reuse"]) < len(kr) or len(vals["new"]) < len(kn) or any(not math.isfinite(v) for v in vals["reuse"] + vals["new"]):
                continue  # a side not (fully) read: the cell is drawn x
            c = {**vals, "n_reuse": len(kr), "n_new": len(kn)}
            PF.score("crosscoder", c)
            out[(pair, col)] = {"gap": c["gap"], "apart": c["call"] == "apart", "call": c["call"], "values": vals,
                                "source": f"experiments/crosscoder/results/sides/<state>.json .d ({col})"}  # fmt: skip
    return out


def main() -> None:
    sides = {p.stem: json.loads(p.read_text()) for p in sorted((RES / "sides").glob("*.json"))}
    m67 = json.loads((REPO / "experiments/size_matched/results/m67.json").read_text())["pairs"]
    table, pairs, names = GP.cells_67m()
    table |= cells(sides, m67)
    for c in json.loads(SMATCH.read_text())["cells"]:
        if c["model"] == "67M" and c["reading"] == "share":
            table[(c["pair"], "share_matched")] = {"gap": c["matched_gap"], "apart": c["matched_call"] == "apart", "source": f"{SMATCH} .cells[share {c['pair']}]"}
    cols: dict[str, str] = {}
    for k, v in GP.COLS.items():
        cols[k] = v
        if k == "share":
            cols |= NEW_COLS
    rows = [{c: table[(p, c)] for c in cols if (p, c) in table} for p in pairs]
    fig, ax = figure((14.0, 7.5))
    GP.draw(ax, rows, [names[p] for p in pairs], cols, "reuse side minus new side")
    ax.set_title("67M scorecard with the crosscoder beside the re-gating share: shared-latent part minus fine-tune-specific part of the activation change", fontsize=10, pad=8)
    data = GP.records("67M", [names[p] for p in pairs], rows, cols)
    built = PF.BUILT
    tally = {col: {"gap > 0": sum(1 for p in built if (p, col) in table and table[(p, col)]["gap"] > 0),
                   "apart": sum(1 for p in built if (p, col) in table and table[(p, col)]["apart"]),
                   "read": sum(1 for p in built if (p, col) in table)} for col in ("crosscoder", "crosscoder_matched")}  # fmt: skip
    out = {"cells": [r for r in data if r["column"] in NEW_COLS], "tally_built": tally, "n_sides": len(sides)}
    (RES / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(tally), flush=True)
    save(fig, HERE / "figures/scorecard_crosscoder.png", data=data,
         n="Seeds 2 to 8 a side (the share's saved fine-tunes); crosscoder cells: one crosscoder per fine-tune, seed 0; box = every seed of one side beyond every seed of the other; x = not run.")


if __name__ == "__main__":
    main()
