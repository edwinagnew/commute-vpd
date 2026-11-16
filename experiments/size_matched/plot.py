"""P2.28's description and figure, from saved results only (no model, CPU): every pair cell's unscaled gap against its
size-matched gap, called by the rule each panel already used, and the outcome per cell. Writes results/summary.json and
figures/size_matched.png (+ .pdf, .csv). Rules: results/predictions.json "rules".

    ~/commute-agents/fleet/cpu-run --mem 6G python experiments/size_matched/plot.py

67M: a cell is (reuse values, new values) as in experiments/method_table/results/final.json; the matched cell keeps the
smaller side's saved values and takes the larger side's from results/m67.json; both are called by the scorecard's own
rule (`plot_final.score`: apart iff the seed ranges do not overlap). Qwen: per seed, side A minus side B; the share by
P3.9's rule (`qwen_library_share` cell, band = |random gap| + spread, the matched random gap from this run's floor),
the overlap and steering by P3.7's (`rules.floored` against P3.7's floor bands). Effective rank and size are not drawn:
the first is unchanged by one scalar (run.py and qwen.py stop otherwise) and the second is matched by construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.figures import METHODS, figure, save
from commute.jobs import load_experiment

HERE = Path(__file__).resolve().parent
RES, REPO = HERE / "results", HERE.parents[1]
PF = load_experiment("method_table", "plot_final")  # BUILT, score: the scorecard's rule
QS = load_experiment("qwen_library_share", "run")  # cell: P3.9's rule
RULES = load_experiment("qwen_library_readings", "rules")  # floored: P3.7's rule
P39 = REPO / "experiments/qwen_library_share/results/p-015e0710"
P37 = REPO / "experiments/qwen_library_readings/results"
SEP = {"apart", "beyond band", "beyond floor"}
KIND = {"last": "late", "5x": "matched"}
M67 = {"share": "regating", "steering": "steering", "neuron_top": "neuron", "k_half": "neuron_half"}
QWEN = {"share": "regating", "overlap": "neuron", "steer": "steering"}


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def outcome(before: str, after: str, g0: float, g1: float) -> str:
    """kept / flipped (separated both times, same or opposite sign), lost, gained, neither; unreadable if either is."""
    if "unreadable" in (before, after) or "one value" in before + after:
        return "unreadable"
    a, b = before in SEP, after in SEP
    if a and b:
        return "kept" if g0 * g1 > 0 else "flipped"
    return "lost" if a else "gained" if b else "neither"


def value(rec: dict[str, Any], col: str, form: str) -> float | None:
    x = {"share": lambda: rec["share"]["r"], "k_half": lambda: rec["k_half"]["k_half"],
         "steering": lambda: rec["steering"].get(form), "neuron_top": lambda: rec["neuron_top"][form]}[col]()  # fmt: skip
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def m67_cells() -> list[dict[str, Any]]:
    final = {(c["pair"], c["method"]): c for c in load(REPO / "experiments/method_table/results/final.json")["cells"]}
    run = load(RES / "m67.json")["pairs"]
    out = []
    for pair in PF.BUILT:
        p = run[pair]
        for col in M67:
            old = dict(final[(pair, col)])
            form = old.get("form", "").split(",")[0]
            got = [value(r, col, form) for r in p["states"].values()]
            new = dict(old, **{p["larger"]: [x for x in got if x is not None], f"n_{p['larger']}": sum(x is not None for x in got)})
            PF.score(col, old)
            PF.score(col, new)
            out.append({"model": "67M", "pair": pair, "reading": col, "form": form, "c": p["c"], "larger": p["larger"],
                        "gap": old["gap"], "call": old["call"], "matched_gap": new["gap"], "matched_call": new["call"],
                        "matched_values": {"reuse": new["reuse"], "new": new["new"]},
                        "outcome": outcome(old["call"], new["call"], old["gap"], new["gap"])})  # fmt: skip
    return out


def share_cells(pair: str, spread: float) -> list[dict[str, Any]]:
    p, summ, out = RULES.PAIRS[pair], load(P39 / "summary.json"), []
    mine = RES / "qwen" / pair / "share"
    for save, kind in KIND.items():
        floor, old = load(mine / "floor" / f"{save}.json"), summ[f"{pair}/{kind}"]
        big = floor["larger"]
        rec = lambda arm, s, save=save, big=big: load(mine / save / f"{arm}_s{s}_{save}.json") if arm == big else load(P39 / "run2" / f"{arm}_s{s}_{save}.json")
        gaps = [rec(p.a, s)["regating"] - rec(p.b, s)["regating"] for s in RULES.SEEDS]
        small_rand = load(P39 / "run1" / f"{floor['smaller']}_s0_{save}_kl.json")["regating"]
        rand = floor["regating"] - small_rand if big == p.a else small_rand - floor["regating"]
        band = abs(rand) + spread
        cell = QS.cell(gaps, band, len(RULES.SEEDS))
        out.append({"model": "Qwen", "pair": pair, "save": save, "reading": "share", "c": floor["c"], "larger": big,
                    "gap": fmean(old["gaps"]), "call": old["cell"], "band": old["band"], "matched_gaps": gaps,
                    "matched_gap": fmean(gaps), "matched_call": cell, "matched_band": band, "matched_random_gap": rand,
                    "outcome": outcome(old["cell"], cell, fmean(old["gaps"]), fmean(gaps))})  # fmt: skip
    return out


def diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def mean_or_nan(xs: list[float | None]) -> float:
    return math.nan if any(x is None for x in xs) else fmean(x for x in xs if x is not None)


def cheap_cells(pair: str) -> list[dict[str, Any]]:
    p, out = RULES.PAIRS[pair], []
    old, band = load(P37 / pair / "stage2.json")["states"], load(P37 / pair / "stage1.json")["band"]
    mine = load(RES / "qwen" / pair / "readings.json")
    for save in KIND:
        big = mine["cells"][save]["larger"]
        for k in ("overlap", "steer"):
            get = lambda arm, s, scaled, k=k, save=save: (mine["states"] if scaled else old)[f"{arm}_s{s}_{save}"][k]
            g0 = [diff(get(p.a, s, False), get(p.b, s, False)) for s in RULES.SEEDS]
            g1 = [diff(get(p.a, s, p.a == big), get(p.b, s, p.b == big)) for s in RULES.SEEDS]
            a, b = RULES.floored(g0, band[k]), RULES.floored(g1, band[k])
            m0, m1 = mean_or_nan(g0), mean_or_nan(g1)
            out.append({"model": "Qwen", "pair": pair, "save": save, "reading": k, "c": mine["cells"][save]["c"], "larger": big,
                        "gap": m0, "call": a, "matched_gaps": g1, "matched_gap": m1, "matched_call": b,
                        "band": band[k], "outcome": outcome(a, b, m0, m1)})  # fmt: skip
    return out


def draw(cells: list[dict[str, Any]]) -> None:
    panels = [("67M", c) for c in M67] + [("Qwen", c) for c in QWEN]
    fig, axes = figure((10.0, 5.2), ncols=4, nrows=2)
    axes.flat[-1].set_visible(len(panels) == axes.size)  # 4 + 3 panels on a 2 x 4 grid: the last one is blank
    for ax, (model, col) in zip(axes.flat, panels):
        rows = [c for c in cells if c["model"] == model and c["reading"] == col]
        st = METHODS[(M67 if model == "67M" else QWEN)[col]]
        for c in rows:
            filled = c["matched_call"] in SEP
            ax.scatter(c["gap"], c["matched_gap"], s=30, marker=st.marker, color=st.color if filled else "white", edgecolors=st.color, linewidths=1)
        xs = [c["gap"] for c in rows] + [c["matched_gap"] for c in rows] or [0.0, 1.0]
        lo, hi = min(xs + [0.0]), max(xs + [0.0])
        ax.plot([lo, hi], [lo, hi], color="#777777", linewidth=0.8, linestyle="--", label="no change with size")
        ax.axhline(0, color="#bbbbbb", linewidth=0.8)
        ax.set_title(f"{model}: {st.label if col != 'k_half' else 'neurons for half the gain'}", fontsize=8)
        ax.set_xlabel("gap, unscaled")
        ax.set_ylabel("gap, sizes matched")
    axes.flat[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Gaps before and after the larger update is shrunk to its partner's size (filled: still separated)")
    n = ("67M: nine built pairs, 2 to 8 seeds a side; Qwen: three pairs at two saves, 3 seeds a side. Gap: side built to "
         "reuse minus side built new (Qwen: side A minus side B); neurons for half the gain in log10 of new over reuse. "
         "On the dashed line size played no part; on 0 the gap was all size.")
    save(fig, HERE / "figures" / "size_matched.png", n=n, data=[{k: v for k, v in c.items() if not isinstance(v, (list, dict))} for c in cells])


def main() -> None:
    spread = load(P39 / "predictions.json")["description"]["spread"]
    cells = m67_cells() + [c for pair in ("copy_cipher", "facts", "gold") for c in share_cells(pair, spread) + cheap_cells(pair)]
    tally = {}
    for c in cells:
        key = f"{c['model']} {c['reading']}"
        tally.setdefault(key, {}).setdefault(c["outcome"], 0)
        tally[key][c["outcome"]] += 1
    (RES / "summary.json").write_text(json.dumps({"cells": cells, "tally": tally}, indent=1))
    for k, v in tally.items():
        print(f"[tally] {k}: {json.dumps(v)}")
    draw(cells)


if __name__ == "__main__":
    main()
