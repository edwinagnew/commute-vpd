"""E_TBD_RUN1 scoring, $0, CPU, saved arrays only (plan reports/plans/2026-11-26_plan_p1-65-readthrough-fills.md, Run 1).
Per fine-tune: E208's saved per-part r_main (rho_saved_items/results/large/parts/<item>.npz, keys parts and r_main) is
mapped by part name ("L<l>.n<i>", "L<l>.h<h>") onto this run's K (results/large/items_<item>.npz: in_K, changed, in
E277's variable order: per layer 6 heads, 3072 neurons, one remainder). Two K values:
    k_value          mean r_main over K and changed (1 when that set is empty, the do-nothing's convention)
    k_value_with_1   mean over all of K, each unchanged member counted at 1
beside E208's K* value (rho_saved_items/results/summary.json R3.<item>.rho_main_replace) and the bounds that need no
run (items/<item>.json r_changed.main: mean and min over E208's changed parts). A K and changed member that E208's parts
do not list is counted (n_missing, expected 0) and left out of both values; a remainder in K is counted (expected 0:
at the width band it writes 0, so its swap changes nothing).

    ~/commute-agents/fleet/cpu-run python3 experiments/p165_cos2_k/score.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from commute import needed_neurons as NN
from commute.jobs import EXPERIMENTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E208 = EXPERIMENTS / "rho_saved_items" / "results"


def names(n: int) -> list[str]:
    """E277's variable names in its order (needed_neurons.variables: heads, neurons, the remainder, per layer)."""
    assert n % NN.PER_LAYER == 0, n
    per = lambda l: [f"L{l}.h{h}" for h in range(NN.H)] + [f"L{l}.n{i}" for i in range(NN.NEUR)] + [f"L{l}.rem"]
    return [x for l in range(n // NN.PER_LAYER) for x in per(l)]


def score_item(item: str, k_dir: Path) -> dict[str, Any]:
    z, p = np.load(k_dir / f"items_{item}.npz"), np.load(E208 / "large" / "parts" / f"{item}.npz")
    inK, ch = z["in_K"].astype(bool), z["changed"].astype(bool)
    nm = names(len(inK))
    r = dict(zip(p["parts"].tolist(), p["r_main"].tolist(), strict=True))
    sel = [nm[j] for j in np.flatnonzero(inK & ch)]
    got = [r[x] for x in sel if x in r]
    n_unch = int((inK & ~ch).sum())
    e = json.loads((E208 / "items" / f"{item}.json").read_text())["r_changed"]["main"]
    out = {"item": item, "size_K": int(inK.sum()), "n_changed": int(ch.sum()), "n_changed_K": len(sel), "n_missing": len(sel) - len(got),
           "n_rem_K": sum(nm[j].endswith(".rem") for j in np.flatnonzero(inK)), "n_e208_parts": len(r),
           "k_value": float(np.mean(got)) if got else 1.0, "k_min": min(got) if got else None,
           "k_value_with_1": (sum(got) + n_unch) / (len(got) + n_unch) if got or n_unch else 1.0,
           "e208_changed_mean": e["mean"], "e208_changed_min": e["min"],
           "kstar_value": json.loads((E208 / "summary.json").read_text())["R3"][item]["rho_main_replace"]}  # fmt: skip
    out["within_bound"] = out["e208_changed_min"] <= out["k_value"] <= 1.0
    return out


def main(k_dir: Path = RESULTS / "large") -> dict[str, Any]:
    items = sorted(p.stem.removeprefix("items_") for p in k_dir.glob("items_*.npz"))
    rows = [score_item(i, k_dir) for i in items]
    out = {"experiment": json.loads((RESULTS / "predictions.json").read_text())["experiment"], "n_items": len(rows),
           "min_k_value": min((r["k_value"] for r in rows), default=None), "min_k_value_with_1": min((r["k_value_with_1"] for r in rows), default=None),
           "min_kstar_value": min((r["kstar_value"] for r in rows), default=None), "all_within_bound": all(r["within_bound"] for r in rows),
           "n_missing": sum(r["n_missing"] for r in rows), "n_rem_K": sum(r["n_rem_K"] for r in rows), "rows": rows}  # fmt: skip
    for r in rows:
        print(f"{r['item']:16s} K {r['k_value']:.6f} (with unchanged at 1 {r['k_value_with_1']:.6f}), K* {r['kstar_value']:.6f}; "
              f"|K| {r['size_K']} ({r['n_changed_K']} changed, {r['n_missing']} not in E208's parts); E208 changed min {r['e208_changed_min']:.4f}")  # fmt: skip
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return out


if __name__ == "__main__":
    s = main()
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=1))
