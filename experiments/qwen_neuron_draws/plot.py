"""P3.15's description and figures, from saved results only (CPU, no model). Writes results/summary.json,
results/size_matched_draws.csv (every value with its source file and key), figures/size_matched_draws.{png,pdf,csv}
and figures/unmatched_last_draws.{png,pdf,csv}. Rules: results/predictions.json "description".

    ~/commute-agents/fleet/cpu-run --mem 4G python experiments/qwen_neuron_draws/plot.py [--results DIR]

Size-matched (E187's comparison): per pair and save, the seed gaps (side A minus side B, seeds 0-2) are E187's, recomputed
from its files as its plot.py `share_cells` does and asserted equal to its summary.json. Each draw's random gap is the
larger side's random change at the scaled seed-0 state minus or plus the smaller side's at its own seed-0 state, signed
A - B: draw 0 from E187's floor and P3.9's run 1; draws 1-2 from this run's d<d>/floor and d<d>/side records.
Unmatched (the paper's qwen_share, frozen-neuron panel): each side's last-save share for seeds 0-2 (P3.9 run 2) and its
random change per draw (draw 0 P3.9 run 1, draws 1-2 this run's side records).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from commute.figures import METHODS, figure, save
from commute.jobs import EXPERIMENTS, load_experiment
from commute.qwen_reading import r9

HERE = Path(__file__).resolve().parent
QS = load_experiment("qwen_library_share", "run")  # cell: P3.9's rule
RULES = load_experiment("qwen_library_readings", "rules")  # PAIRS: side A, side B
P39 = EXPERIMENTS / "qwen_library_share" / "results" / "p-015e0710"
E187 = EXPERIMENTS / "size_matched" / "results"
PAIRS = tuple(QS.PAIRS)  # copy_cipher, facts, gold (RULES.PAIRS also holds template, which has no 5x save)
SAVES, DRAWS, SEEDS = ("last", "5x"), (0, 1, 2), RULES.SEEDS
assert all(QS.PAIRS[p] == (RULES.PAIRS[p].a, RULES.PAIRS[p].b) for p in PAIRS)
KIND = {"last": "late", "5x": "matched"}
SIDE_NAMES = {"copy": "copy", "cipher": "cipher", "known": "known facts", "invented": "invented facts",
              "gold": "maths, real solutions", "gold_shuf": "maths, shuffled solutions"}


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def rel(p: Path) -> str:
    return str(p.relative_to(EXPERIMENTS.parent) if p.is_relative_to(EXPERIMENTS.parent) else p)


def share(p: Path) -> float | None:
    """The record's share, None unless 'scored' (an unscored draw is reported, not used)."""
    rec = load(p)
    return float(rec["regating"]) if rec["status"] == "scored" else None


def draw_range(seed_gaps: list[float], random_gaps: list[float | None]) -> dict[str, Any]:
    """The three-draw description: the range of the scored random gaps, and whether every seed gap lies strictly
    outside it on one side (rounded to 9 places, as P3.9's cell). No scored draw: 'unreadable'."""
    rs = [r9(g) for g in random_gaps if g is not None]
    if not rs:
        return {"range": None, "n_draws": 0, "seeds_vs_range": "unreadable"}
    lo, hi, gs = min(rs), max(rs), [r9(g) for g in seed_gaps]
    where = "all above" if all(g > hi for g in gs) else "all below" if all(g < lo for g in gs) else "not all outside"
    return {"range": [lo, hi], "n_draws": len(rs), "seeds_vs_range": where}


def random_files(res: Path, pair: str, save: str, arm: str, big: str, d: int) -> Path:
    """The record of `arm`'s random change at draw d: the larger side's at its scaled state, the smaller side's at its own."""
    if d == 0:
        return E187 / "qwen" / pair / "share" / "floor" / f"{save}.json" if arm == big else P39 / "run1" / f"{arm}_s0_{save}_kl.json"
    return res / pair / f"d{d}" / (f"floor/{save}.json" if arm == big else f"side/{arm}_s0_{save}_kl.json")


def matched(res: Path, pair: str, save: str, spread: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    p = RULES.PAIRS[pair]
    floor = load(E187 / "qwen" / pair / "share" / "floor" / f"{save}.json")
    big = floor["larger"]
    state = lambda arm, s: (E187 / "qwen" / pair / "share" / save if arm == big else P39 / "run2") / f"{arm}_s{s}_{save}.json"
    gaps: list[float] = []
    for s in SEEDS:
        fa, fb = state(p.a, s), state(p.b, s)
        gaps.append(load(fa)["regating"] - load(fb)["regating"])
        rows.append({"block": "size_matched", "pair": pair, "save": save, "what": "seed_gap", "seed": s, "draw": "",
                     "value": gaps[-1], "key": "regating(a) - regating(b)", "source_a": rel(fa), "source_b": rel(fb)})
    e187 = next(c for c in load(E187 / "summary.json")["cells"] if c.get("pair") == pair and c.get("save") == save and c["reading"] == "share")
    assert gaps == e187["matched_gaps"], (pair, save, gaps, e187["matched_gaps"])
    rand: dict[int, float | None] = {}
    bands: dict[int, float | None] = {}
    cells: dict[int, str] = {}
    for d in DRAWS:
        fa, fb = random_files(res, pair, save, p.a, big, d), random_files(res, pair, save, p.b, big, d)
        a, b = share(fa), share(fb)
        rand[d] = None if a is None or b is None else a - b
        bands[d] = None if rand[d] is None else abs(rand[d]) + spread
        cells[d] = QS.cell(gaps, bands[d], len(SEEDS))
        rows.append({"block": "size_matched", "pair": pair, "save": save, "what": "random_gap", "seed": 0, "draw": d,
                     "value": rand[d], "key": "regating(a) - regating(b), None unless both scored", "source_a": rel(fa), "source_b": rel(fb)})
        rows.append({"block": "size_matched", "pair": pair, "save": save, "what": "band", "seed": "", "draw": d, "value": bands[d],
                     "key": f"|random gap| + spread ({spread:.6g})", "source_a": rel(fa), "source_b": rel(fb)})
    assert rand[0] == e187["matched_random_gap"], (pair, save, rand[0], e187["matched_random_gap"])
    return {"pair": pair, "save": save, "kind": KIND[save], "a": p.a, "b": p.b, "larger": big, "c": floor["c"],
            "seed_gaps": gaps, "random_gaps": rand, "bands": bands, "cell_per_draw": cells,
            **draw_range(gaps, list(rand.values()))}


def unmatched(res: Path, pair: str, arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds: list[float] = []
    rand: dict[int, float | None] = {}
    for s in SEEDS:
        f = P39 / "run2" / f"{arm}_s{s}_last.json"
        seeds.append(load(f)["regating"])
        rows.append({"block": "unmatched_last", "pair": pair, "save": "last", "what": f"share {arm}", "seed": s, "draw": "",
                     "value": seeds[-1], "key": "regating", "source_a": rel(f), "source_b": ""})
    for d in DRAWS:
        f = random_files(res, pair, "last", arm, "", d)  # never the larger side's: each side at its own state
        rand[d] = share(f)
        rows.append({"block": "unmatched_last", "pair": pair, "save": "last", "what": f"random share {arm}", "seed": 0, "draw": d,
                     "value": rand[d], "key": "regating, None unless scored", "source_a": rel(f), "source_b": ""})
    return {"pair": pair, "arm": arm, "seeds": seeds, "random": rand}


def draw_matched(cells: list[dict[str, Any]], out: Path) -> None:
    fig, axes = figure((7.5, 3.0), ncols=3)
    ft, rd = METHODS["regating"], METHODS["random"]
    for ax, pair in zip(axes, PAIRS):
        for i, c in enumerate(x for x in cells if x["pair"] == pair):
            ax.scatter([i - 0.15] * len(c["seed_gaps"]), c["seed_gaps"], color=ft.color, marker=ft.marker, s=20,
                       label="fine-tune, seeds 0-2" if i == 0 else None)
            for j, d in enumerate(DRAWS):
                g, b = c["random_gaps"][d], c["bands"][d]
                if g is None or b is None:
                    continue
                x = i + 0.1 + 0.08 * j
                ax.vlines(x, -b, b, color="#bbbbbb", linewidth=2, label="band, |random gap| + spread" if i == j == 0 else None)
                ax.scatter([x], [g], color=rd.color, marker=rd.marker, s=24, label="random change, draws 0-2" if i == j == 0 else None)
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.set_xticks(range(len(SAVES)), [f"{s} (c {c['c']:.2f})" for s, c in zip(SAVES, (x for x in cells if x["pair"] == pair))])
        p = RULES.PAIRS[pair]
        ax.set_title(f"{SIDE_NAMES[p.a]} minus {SIDE_NAMES[p.b]}", fontsize=8)
        ax.set_xlabel("save")
    axes[0].set_ylabel("share gap, sizes matched")
    axes[0].legend(fontsize=6.5, frameon=False, loc="best")
    fig.suptitle("Frozen-neuron share gaps at matched update size against three random draws per side", fontsize=9)
    n = ("p-015e0710; per pair and save the larger side's three states are scaled by c to the smaller side's mean |dW| "
         "(E187). Fine-tune: 3 seeds, each a side A minus side B gap. Random: 3 Gaussian draws per side, KL-matched on 16 "
         "rows, read on 128 rows of 256 tokens.")
    save(fig, out / "size_matched_draws.png", n=n, data=[{k: v for k, v in c.items() if not isinstance(v, (list, dict))} | {
        **{f"seed_gap_{s}": g for s, g in zip(SEEDS, c["seed_gaps"])}, **{f"random_gap_d{d}": c["random_gaps"][d] for d in DRAWS},
        **{f"band_d{d}": c["bands"][d] for d in DRAWS}} for c in cells])  # fmt: skip


def draw_unmatched(sides: list[dict[str, Any]], out: Path) -> None:
    fig, ax = figure((5.0, 2.6))
    ft, rd = METHODS["regating"], METHODS["random"]
    for y, s in enumerate(sides):
        ax.scatter(s["seeds"], [y] * len(s["seeds"]), color=ft.color, marker=ft.marker, s=16, alpha=0.8, label="fine-tune, seeds 0-2" if y == 0 else None)
        shown = [v for v in s["random"].values() if v is not None]
        ax.scatter(shown, [y] * len(shown), color=rd.color, marker=rd.marker, s=22, alpha=0.7, label="random change, same KL, draws 0-2" if y == 0 else None)
    for y in (1.5, 3.5):
        ax.axhline(y, color="#DDDDDD", linewidth=0.6)
    xs = [v for s in sides for v in [*s["seeds"], *s["random"].values()] if v is not None]
    ax.set_xlim(min(0.78, min(xs) - 0.01), max(1.03, max(xs) + 0.01))  # the paper's qwen_share range, widened to the data
    ax.set_yticks(range(len(sides)), [SIDE_NAMES[s["arm"]] for s in sides], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("re-gating share, last save")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.set_title("Frozen-neuron decomposition: each side's share and three random draws", fontsize=8)
    save(fig, out / "unmatched_last_draws.png", n="p-015e0710; 3 seeds per side (P3.9 run 2); 3 random draws per side, seed-0 state's norms, KL-matched on 16 rows",
         data=[{"arm": s["arm"], **{f"seed_{i}": v for i, v in enumerate(s["seeds"])}, **{f"random_d{d}": v for d, v in s["random"].items()}} for s in sides])


def main() -> None:
    ap = argparse.ArgumentParser(description="P3.15's description and figures")
    ap.add_argument("--results", type=Path, default=HERE / "results", help="the run's records (results/<pair>/d<d>/...)")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    spread = load(P39 / "predictions.json")["description"]["spread"]
    rows: list[dict[str, Any]] = []
    cells = [matched(a.results, pair, save, spread, rows) for pair in PAIRS for save in SAVES]
    sides = [unmatched(a.results, pair, arm, rows) for pair in PAIRS for arm in QS.PAIRS[pair]]
    (a.results / "summary.json").write_text(json.dumps({"size_matched": cells, "unmatched_last": sides}, indent=1))
    with (a.results / "size_matched_draws.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for c in cells:
        print(f"[matched] {c['pair']} {c['save']}: seeds {[round(g, 5) for g in c['seed_gaps']]}, random "
              f"{ {d: None if g is None else round(g, 5) for d, g in c['random_gaps'].items()} }, {c['seeds_vs_range']}, "
              f"cells {c['cell_per_draw']}, mean seed gap {fmean(c['seed_gaps']):.5f}")
    draw_matched(cells, a.figures)
    draw_unmatched(sides, a.figures)


if __name__ == "__main__":
    main()
