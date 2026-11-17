"""E156's figure and outcome (cell P1.13), from results/{control,ft_s0,ft_s1}.json and results/predictions.json only (no
model): the best found H per measurement, both seeds, against the H cut-off 0.5 and the ceiling 1 - TV(ft, F) / TV(ft, M), as
the card's sketch (figures/sketch.png). Also writes results/readings.json: per seed the best H of each variant and its rise
over E154's best found H, and the outcome by the filed rule (predictions.json, "outcome_rule").

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/h_diagnosis/plot.py     (after the job)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1)
CUT_H = 0.5  # decisions.md, 2026-11-10 (working definition adopted tentatively); predictions.json must agree
VARIANTS = {"c": "(c) ridge penalty down to 1e-8", "a": "(a) map written at every position", "b": "(b) two sites, in order"}
EXPECTED = {"c": 4, "a": 3, "b": 10}  # readings per seed when a variant is complete (lams_c, all_sites, pairs in run.Config)


def best_h(rec: dict[str, Any]) -> float | None:
    """The largest H among a variant's readings; None if every reading failed (a factorisation at a tiny penalty)."""
    hs = [v["H"] for v in rec.values() if v.get("H") is not None]
    return max(hs) if hs else None


def outcome(h_ctrl: float | None, raises: dict[str, list[float | None]], complete: bool, th: dict[str, Any]) -> str:
    """The card's rule. h_ctrl: H on the exact target F at in2. raises: per variant, per seed, its best H minus E154's best
    (None: no reading, which counts as no rise). Rival a needs one variant raising both seeds; rival b, none raising either."""
    if h_ctrl is None:
        return "not read"
    if h_ctrl < th["stop_bar"]:
        return "rival b (the card's stop): the search cannot read an affine edit it should read perfectly"
    if not complete:
        return "incomplete: resubmit the job"
    if h_ctrl < th["control_bar"]:
        return "no call: the control reads between the stop and its bar"
    raised = {v: [x is not None and x >= th["rise"] for x in xs] for v, xs in raises.items()}
    if any(all(f) for f in raised.values()):
        return "rival a: the change is partly out of the map's reach, and " + ", ".join(VARIANTS[v] for v, f in raised.items() if all(f)) + " reaches it"
    if not any(any(f) for f in raised.values()):
        return "rival b: the control passes and no variant raises H by the rise bar on either seed"
    return "no call: a rise on one seed, but no variant raises both"


def readings(results: Path, th: dict[str, Any]) -> dict[str, Any]:
    ctrl = json.loads((results / "control.json").read_text()) if (results / "control.json").is_file() else {}
    h_ctrl = ctrl.get("H", {}).get("in2", {}).get("H")
    items, complete = [], True
    for s in SEEDS:
        path = results / f"ft_s{s}.json"
        out = json.loads(path.read_text()) if path.is_file() else {}
        row: dict[str, Any] = {"seed": s, "e154_best": out.get("e154_best"), "e154_site": out.get("e154_best_site"), "ceiling": out.get("ceiling")}
        for v in VARIANTS:
            rec = out.get(v, {})
            complete = complete and len(rec) == EXPECTED[v] and row["e154_best"] is not None
            b = best_h(rec)
            row[v] = b
            row[f"rise_{v}"] = None if b is None or row["e154_best"] is None else b - row["e154_best"]
        items.append(row)
    raises = {v: [r[f"rise_{v}"] for r in items] for v in VARIANTS}
    return {"control_H_in2": h_ctrl, "control_H": {k: v["H"] for k, v in ctrl.get("H", {}).items()}, "items": items,
            "complete": complete, "outcome": outcome(h_ctrl, raises, complete, th)}


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    pred = json.loads((results / "predictions.json").read_text())
    assert pred["cutoffs"]["H"] == CUT_H, pred["cutoffs"]
    read = readings(results, pred["thresholds"])
    (results / "readings.json").write_text(json.dumps(read, indent=1))
    print(json.dumps(read, indent=1))
    rows = read["items"]
    groups = {"H on the exact target F at in2\n(positive control, one target)": [read["control_H_in2"]] if read["control_H_in2"] is not None else [],
              "fine-tune as E154 read it": [r["e154_best"] for r in rows if r["e154_best"] is not None],
              **{name: [r[v] for r in rows if r[v] is not None] for v, name in VARIANTS.items()}}
    fig, ax = FG.figure(FG.ONE_COLUMN)
    FG.dots(ax, groups, method="regating", xlabel="best found H (each seed of the fine-tune)")
    FG.reference(ax, CUT_H, f"cut-off {CUT_H}")
    ceilings = [r["ceiling"] for r in rows if r["ceiling"] is not None]
    if ceilings:
        FG.reference(ax, sum(ceilings) / len(ceilings), f"ceiling {sum(ceilings) / len(ceilings):.2f}")  # per seed in the .csv
    ax.set_xlim(min([-0.1, *[h for hs in groups.values() for h in hs]]) - 0.05, 1.05)
    ax.set_title(f"Outcome: {read['outcome']}", fontsize=7, loc="left", pad=12, wrap=True)
    fig.suptitle("Why does H stop near 0.47 on the no-teaching control?", fontsize=9)
    FG.save(fig, out / "h_diagnosis.png", n="2 seeds (E154's saved fine-tunes); d on 100,000 read windows (held-out number pairs "
            "and far windows); the control is F itself, one target", data=[{"control_H_in2": read["control_H_in2"]}, *rows])


if __name__ == "__main__":
    main()
