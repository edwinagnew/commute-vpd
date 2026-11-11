"""Cell P2.20: part 1's within-pair order (the found re-weighting share G) against every part 2 method's order, on the
pairs both read on the same saved updates. Saved results only: no model, no GPU. The rule is results/predictions.json.

Overlap is checked by path: each part 1 state file (commute.old_tasks.TASKS, symlinks resolved) must be among the files
E136's inventory lists for the method table's pair, on the same side. The inventory is the file list that the method
table's share, untrained, neuron-top and steering sources read; the other sources were traced by hand (see the P2.20 page).

Usage (repo root):  ~/commute-agents/fleet/cpu-run --mem 6G python experiments/part1_vs_methods/run.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commute.jobs import load_experiment
from commute.old_tasks import TASKS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LABELS = REPO / "experiments/old_task_labels/results/labels_cut_0p8_0p5.json"
FINAL = REPO / "experiments/method_table/results/final.json"
INVENTORY = REPO / "experiments/share_untrained/results/inventory.json"
PAIRS = load_experiment("old_task_labels", "plot_gate").PAIRS  # row label -> (reuse task, new task)
# part 1 pair (reuse task) -> (final.json pair, E136 inventory pair id)
MATCH = {"e62_copy": ("e62_copy_lookup", "e62_copy_lookup"), "copy24": ("open_pair_c3_copy", "open_pair_c3_copy"),
         "ident32": ("e46_copy_random", "e46_copy_random"), "case_real": ("e123_case_label", "case_label"),
         "repeat_real": ("e125_repeat_label", "fresh_reuse_pair")}  # fmt: skip
# Same task family, other updates: listed, not scored.
TASK_ONLY = {"copy24": ("e58_copy_prior_cipher", "96 keys, prior-matched cipher, refit states in experiments/refit_96key/results/large")}
METHODS = ("share", "prior", "steering", "erank", "size", "neuron", "neuron_half", "neuron_top", "untrained")
SEEDS = (0, 1)


def part1() -> dict[str, dict[str, Any]]:
    """Per pair: G per side and seed, the gap per seed (reuse minus new) and its mean."""
    g = {it["item"]: it["G"] for it in json.loads(LABELS.read_text())["items"]}
    out = {}
    for label, (a, b) in PAIRS.items():
        per_seed = [g[f"{a}_s{s}"] - g[f"{b}_s{s}"] for s in SEEDS]
        out[a] = {"label": label, "reuse": a, "new": b, "G_reuse": [g[f"{a}_s{s}"] for s in SEEDS],
                  "G_new": [g[f"{b}_s{s}"] for s in SEEDS], "gap_seeds": per_seed, "gap": sum(per_seed) / len(per_seed)}  # fmt: skip
    return out


def overlap(reuse: str, new: str, inv_id: str) -> dict[str, Any]:
    """Whether every part 1 state file of each side is among the inventory's files of that side, by resolved path."""
    inv = json.loads(INVENTORY.read_text())
    pair = next(p for p in inv["pairs"] if p["id"] == inv_id)
    out: dict[str, Any] = {"inventory_pair": inv_id}
    for side, task in (("reuse", reuse), ("new", new)):
        theirs = {(REPO / inv["states"][s]["file"]).resolve(): s for s in pair[side]}
        files = list(TASKS[task].states.values())
        out[side] = {"part1_files": [str(p.relative_to(REPO)) for p in files],
                     "matched_states": [theirs.get(p.resolve()) for p in files], "part2_states": len(pair[side])}  # fmt: skip
    out["same_files"] = all(None not in out[s]["matched_states"] for s in ("reuse", "new"))
    return out


def order(cell: dict[str, Any] | None, p1_gap: float) -> str:
    """The filed method_order_rule."""
    if cell is None:
        return "missing"
    if cell["call"] != "apart":
        return "no order"
    return "agree" if (cell["gap"] > 0) == (p1_gap > 0) else "reverse"


def ranks(x: list[float]) -> list[float]:
    """Average ranks, ties shared."""
    srt = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and x[srt[j + 1]] == x[srt[i]]:
            j += 1
        for k in range(i, j + 1):
            r[srt[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation of the average ranks; None below 3 points or with a constant side."""
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / (sxx * syy) ** 0.5


def main() -> None:
    p1 = part1()
    cells = {(c["pair"], c["method"]): c for c in json.loads(FINAL.read_text())["cells"]}
    pairs = {a: {**p1[a], "final_pair": MATCH[a][0], "overlap": overlap(a, p1[a]["new"], MATCH[a][1])} for a in MATCH}
    scored = [a for a in pairs if pairs[a]["overlap"]["same_files"]]
    methods: dict[str, Any] = {}
    for m in METHODS:
        rows = {a: cells.get((pairs[a]["final_pair"], m)) for a in scored}
        per = {a: {"order": order(c, p1[a]["gap"]), "call": None if c is None else c["call"], "gap": None if c is None else c["gap"],
                   "n_reuse": None if c is None else c["n_reuse"], "n_new": None if c is None else c["n_new"]}
               for a, c in rows.items()}  # fmt: skip
        have = [a for a in scored if rows[a] is not None]
        counts = {k: sum(v["order"] == k for v in per.values()) for k in ("agree", "reverse", "no order", "missing")}
        rho = spearman([p1[a]["gap"] for a in have], [per[a]["gap"] for a in have])
        methods[m] = {"pairs": per, "counts": counts, "spearman": rho, "n": len(have)}
    out = {"rule": "experiments/part1_vs_methods/results/predictions.json", "part1": p1, "overlap_count": len(scored),
           "overlapping": pairs, "task_only": {a: {"final_pair": fp, "why_not_scored": why} for a, (fp, why) in TASK_ONLY.items()},
           "methods": methods,
           "missing": [{"pair": pairs[a]["final_pair"], "method": m} for m in METHODS for a in scored if methods[m]["pairs"][a]["order"] == "missing"]}  # fmt: skip
    (HERE / "results" / "compare.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"overlapping pairs: {len(scored)} of {len(pairs)}")
    for m, r in methods.items():
        print(f"{m:12s} {r['counts']} rho {r['spearman']} n {r['n']}")


if __name__ == "__main__":
    main()
