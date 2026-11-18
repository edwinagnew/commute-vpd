"""E203's scorecard columns, from saved results only (CPU, after run.py stage 2): the crosscoder reading per built pair,
as built and size-matched, in two forms, beside the re-gating share on the part 2 scorecard's 67M panel
(experiments/method_table/gate_plot_1116.py), same colour scale, same gap rule (`plot_final.score`). Descriptive: no
reuse/new verdict. Rules: results/run2/predictions.json "rules".

- d, the authors' form: the change's shared part minus its fine-tune-specific part (the side JSON's `d`).
- d_alt: shared part minus (fine-tune-specific + fine-tune artefact) parts, the artefact class counted as fine-tune-only
  (user, 2026-11-18).

One crosscoder per side (each side's first state, `run.py stage2_sides`), so every cell's call is 'one value a side'.

Writes results/run2/summary.json and figures/scorecard_crosscoder.png (+ .pdf, .csv).

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
RES, REPO = HERE / "results/run2", HERE.parents[1]
GP = load_experiment("method_table", "gate_plot_1116")  # cells_67m, draw, records, COLS: the scorecard's 67M panel
PF = load_experiment("method_table", "plot_final")  # score: the scorecard's rule
SM = load_experiment("size_matched", "run")  # pairs(): the share's states per pair
SMATCH = REPO / "experiments/size_matched/results/summary.json"
FORMS = {"d": lambda s: s["d"], "d_alt": lambda s: s["parts"]["shared"] - s["parts"]["ft_specific"] - s["parts"]["ft_artefact"]}
NEW_COLS = {"share_matched": "re-gating\nshare,\nsize-\nmatched",
            "crosscoder": "cross-\ncoder,\nauthors'\nrule", "crosscoder_matched": "cross-\ncoder,\nauthors',\nsize-\nmatched",
            "crosscoder_alt": "cross-\ncoder,\nartefacts\nas fine-\ntune-only", "crosscoder_alt_matched": "cross-\ncoder,\nartefacts,\nsize-\nmatched"}  # fmt: skip


def key(state: str) -> str:
    return state.replace("/", "__")


def side_keys(pair: str, r: str, n: str, larger: str, matched: bool) -> tuple[str, str]:
    """The side JSON stems of one pair: each side's first state; size-matched, the larger side scaled by the pair's c."""
    m = f"__matched_{pair}" if matched else ""
    return key(r) + (m if larger == "reuse" else ""), key(n) + (m if larger == "new" else "")


def cells(sides: dict[str, dict[str, Any]], m67: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per built pair, the crosscoder cells: two forms of d, as built and size-matched."""
    names, _, _ = SM.pairs()
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, (r, n) in names.items():
        for matched in (False, True):
            kr, kn = side_keys(pair, r[0], n[0], m67[pair]["larger"], matched)
            for form, get in FORMS.items():
                col = "crosscoder" + ("_alt" if form == "d_alt" else "") + ("_matched" if matched else "")
                if kr not in sides or kn not in sides or not all(math.isfinite(get(sides[k])) for k in (kr, kn)):
                    continue  # a side not read: the cell is drawn x
                c = {"reuse": [get(sides[kr])], "new": [get(sides[kn])], "n_reuse": 1, "n_new": 1}
                PF.score(col, c)
                out[(pair, col)] = {"gap": c["gap"], "apart": c["call"] == "apart", "call": c["call"], "values": {"reuse": c["reuse"], "new": c["new"]},
                                    "sides": {"reuse": kr, "new": kn}, "source": f"experiments/crosscoder/results/run2/sides/<side>.json {form}"}  # fmt: skip
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
    fig, ax = figure((19.0, 8.5))
    GP.draw(ax, rows, [names[p] for p in pairs], cols, "reuse side minus new side")
    ax.set_title("67M scorecard, crosscoder beside the re-gating share: does the shared-latent part of the activation change read higher"
                 " on the reuse side?\nCrosscoder: higher on 4 to 6 of the 9 built pairs, depending on form; re-gating share: on all 9", fontsize=10, pad=8)  # fmt: skip
    data = GP.records("67M", [names[p] for p in pairs], rows, cols)
    xcols = [c for c in NEW_COLS if c.startswith("crosscoder")]
    tally = {col: {"gap > 0": sum(1 for p in PF.BUILT if (p, col) in table and table[(p, col)]["gap"] > 0),
                   "read": sum(1 for p in PF.BUILT if (p, col) in table)} for col in xcols}  # fmt: skip
    out = {"cells": [r for r in data if r["column"] in xcols], "tally_built": tally, "n_sides": len(sides)}
    (RES / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(tally), flush=True)
    for r in out["cells"]:
        print(r["row"], r["column"], round(r["gap"], 4), r["values"], flush=True)
    save(fig, HERE / "figures/scorecard_crosscoder.png", data=data,
         n="Re-gating and other columns: seeds 2 to 8 a side (box = every seed of one side beyond every seed of the other). Crosscoder columns: one crosscoder per side, so none is boxed; "
           "a reseed of one crosscoder moved d by about 0.05. Planted control missed (latent scaling score 0.244 vs bar 0.2). x = not run.")  # fmt: skip


if __name__ == "__main__":
    main()
