"""E157's figure and reading (cell P1.14), from the results JSONs only (no model): per reading kind (one site, the best of
five sites; two sites, the best of ten pairs in order), the best found H of the planted ceiling, the saved control (E154's
one-site and E156's two-site readings, both seeds, not re-read), the new sum-parity fine-tune (both seeds) and the planted
floor, against the H cut-off 0.5, as the card's sketch (figures/sketch.png). Writes results/readings.json: raw and rescaled
values per row and kind, and the reading per kind by the filed rule (predictions.json, "outcome_rule"). No label.

    ~/commute-agents/fleet/cpu-run --mem 2G python experiments/sum_parity_beside/plot.py     (after the job)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute import figures as FG

HERE = Path(__file__).resolve().parent
E154 = HERE.parent / "built_control_sum" / "results"  # amend1_control_s{0,1}.json: the control's one-site H
E156 = HERE.parent / "h_diagnosis" / "results"  # ft_s{0,1}.json: the control's two-site H (b)
SEEDS = (0, 1)
CUT_H = 0.5  # decisions.md, 2026-11-10; predictions.json must agree
KINDS = {"one_site": ("H", 5, "one site (the definition's measure)"), "two_sites": ("b", 10, "two sites, in order")}
ROWS = {"ceiling": "planted ceiling", "control": "control: last number odd\n(saved, E154/E156)",
        "new": "new: sum parity", "floor": "planted floor"}
EPS = 1e-9  # float rounding at a bar: a difference that is the bar in decimals counts as at the bar


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def best(out: dict[str, Any], kind: str) -> float | None:
    """The largest H of the kind's readings; None unless all of them are there."""
    key, n, _ = KINDS[kind]
    hs = [v["H"] for v in out.get(key, {}).values() if v.get("H") is not None]
    return max(hs) if len(out.get(key, {})) == n and hs else None


def rescale(h: float | None, ceiling: float | None, floor: float | None, th: dict[str, Any]) -> float | None:
    """(h - floor) / (ceiling - floor), or None when a value is missing or ceiling - floor is under scale_min."""
    if h is None or ceiling is None or floor is None or ceiling - floor < th["scale_min"] - EPS:
        return None
    return (h - floor) / (ceiling - floor)


def outcome(new: list[float | None], ctrl: list[float], stopped: bool, complete: bool, th: dict[str, Any]) -> str:
    """The filed reading for one kind. new: the new arm's best H per seed (None: no reading); ctrl: the saved control's."""
    if stopped:
        return "ceiling stop: the search cannot read an edit it can reach exactly; nothing else read"
    if not complete:
        return "incomplete: resubmit the job"
    if not new or any(h is None for h in new):
        return "no reading: a seed of the new arm was not learned, read under the gap floor, or not run"
    hs = [h for h in new if h is not None]
    sep = th["separate"]
    if all(h <= min(ctrl) - sep + EPS for h in hs):
        return "clearly below"
    diff = sum(hs) / len(hs) - sum(ctrl) / len(ctrl)
    if abs(diff) < sep - EPS:
        return "not separated"
    return f"between (new minus control {diff:+.3f})"


def new_seeds(results: Path) -> tuple[list[dict[str, Any]], bool]:
    """The new arm's records per seed, and whether every seed is settled (read in full, or with no reading)."""
    outs = [load(results / f"plain_s{s}.json") for s in SEEDS]
    s0_unlearned = bool(outs[0].get("no_reading")) and "rung" not in outs[0]
    settled = [bool(o.get("no_reading")) or bool(o.get("h_done") and o.get("b_done")) or (s == 1 and not o and s0_unlearned)
               for s, o in zip(SEEDS, outs, strict=True)]
    return outs, all(settled)


def readings(results: Path, th: dict[str, Any]) -> dict[str, Any]:
    ceil, floor, rep = load(results / "ceiling.json"), load(results / "floor.json"), load(results / "repeat.json")
    outs, seeds_done = new_seeds(results)
    stopped = bool(ceil.get("card_stop"))
    complete = all(o.get("h_done") and o.get("b_done") for o in (ceil, floor)) and bool(rep.get("passed")) and seeds_done
    saved = {"one_site": [load(E154 / f"amend1_control_s{s}.json") for s in SEEDS], "two_sites": [load(E156 / f"ft_s{s}.json") for s in SEEDS]}
    kinds: dict[str, Any] = {}
    for kind in KINDS:
        ctrl = [best(o, kind) for o in saved[kind]]
        assert all(h is not None for h in ctrl), f"the saved control's {kind} readings are missing"
        raw = {"ceiling": [best(ceil, kind)], "control": ctrl, "new": [None if o.get("no_reading") else best(o, kind) for o in outs],
               "floor": [best(floor, kind)]}
        hc, hf = raw["ceiling"][0], raw["floor"][0]
        kinds[kind] = {"raw": raw, "span": None if hc is None or hf is None else hc - hf,
                       "rescaled": {r: [rescale(h, hc, hf, th) for h in v] for r, v in raw.items()},
                       "reading": outcome(raw["new"], [h for h in ctrl if h is not None], stopped, complete, th)}
    return {"P1_ceiling_in2": ceil.get("H", {}).get("in2", {}).get("H"), "card_stop": ceil.get("card_stop"),
            "repeat": {"H": rep.get("H"), "H_e156": rep.get("H_e156"), "passed": rep.get("passed")},
            "new_no_reading": [o.get("no_reading") for o in outs], "complete": complete, "kinds": kinds}


def main(results: Path = HERE / "results", out: Path = HERE / "figures") -> None:
    pred = json.loads((HERE / "results" / "predictions.json").read_text())
    assert pred["cutoffs"]["H"] == CUT_H, pred["cutoffs"]
    read = readings(results, pred["thresholds"])
    (results / "readings.json").write_text(json.dumps(read, indent=1))
    print(json.dumps(read, indent=1))
    fig, axes = FG.figure(FG.FULL_WIDTH, ncols=2)
    data: list[dict[str, Any]] = []
    hs_all: list[float] = []
    for ax, (kind, (_, _, name)) in zip(axes, KINDS.items(), strict=True):
        k = read["kinds"][kind]
        groups = {label: [h for h in k["raw"][r] if h is not None] for r, label in ROWS.items()}
        hs_all += [h for hs in groups.values() for h in hs]
        FG.dots(ax, groups, method="regating", xlabel="best found rewiring share (dots: seeds)")
        FG.reference(ax, CUT_H, f"cut-off {CUT_H}")
        ax.set_title(f"{name}\nreading: {k['reading']}", fontsize=7, loc="left", pad=12, wrap=True)
        for r, hs in k["raw"].items():
            data += [{"kind": kind, "row": r, "seed": i if len(hs) > 1 else "", "H": h, "rescaled": k["rescaled"][r][i]} for i, h in enumerate(hs)]
    for ax in axes:
        ax.set_xlim(min([-0.1, *hs_all]) - 0.05, 1.05)
    fig.suptitle("Does the rewiring share read the sum-parity fine-tune below the last-number-odd control?", fontsize=9)
    FG.save(fig, out / "sum_parity_beside.png", n="2 seeds per fine-tune (control: E154's saved fine-tunes, not re-read); one "
            "planted target each for ceiling and floor; d on 100,000 read windows (held-out number pairs and far windows)", data=data)


if __name__ == "__main__":
    main()
