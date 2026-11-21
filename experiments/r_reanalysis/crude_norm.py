"""P1.42 side output: the crude size-normalised R, (1 - R) / size^2, from saved files only (user, 2026-11-20).

    ~/commute-agents/fleet/cpu-run --mem 4G python experiments/r_reanalysis/crude_norm.py

Lower = more reuse. Also 1 - (1 - R) / size^2 (the user's "1 - ..." form; higher = more reuse).
Writes results/crude_norm.json.
"""

from __future__ import annotations

import json
from statistics import mean

import numpy as np
from scipy.stats import rankdata

from commute.jobs import load_experiment

rr = load_experiment("r_reanalysis", "run")


def spearman(a: list[float], b: list[float]) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def main() -> int:
    pairs = {p: v for p, v in rr.load(rr.PAIRS_JSON)["pairs"].items() if p.startswith("e187_")}
    trained = sorted({i for v in pairs.values() for i in (*v["reuse"], *v["new"])})
    items = [*rr.KNOWN, *trained]
    R = {i: rr.load(rr.VPD / "summary.json")["per_item"][i]["R_read"] for i in items}
    size = rr.sizes(items)
    crude = {i: (1 - R[i]) / size[i] ** 2 for i in items}
    known = {i: crude[i] for i in rr.KNOWN}
    gaps = {p: mean(crude[i] for i in v["reuse"]) - mean(crude[i] for i in v["new"]) for p, v in pairs.items()}
    out = {
        "experiment": "P1.42 r_reanalysis crude_norm",
        "definition": "(1 - R_read) / size^2, size = ||dW|| / ||W0|| over the 24 matrices; lower = more reuse",
        "known": {"values": known, "one_minus": {i: 1 - x for i, x in known.items()},
                  "R": {i: R[i] for i in rr.KNOWN}, "size": {i: size[i] for i in rr.KNOWN},
                  **rr.orders(known, higher_is_reuse=False)},  # fmt: skip
        "pairs": {"gaps_reuse_minus_new": gaps, "reuse_lower": sum(g < 0 for g in gaps.values()), "n": len(gaps),
                  "spearman_gap_vs_smaller_update": spearman(list(gaps.values()), [pairs[p]["smaller_update"] for p in gaps])},  # fmt: skip
        "states": {"spearman_crude_vs_size": spearman([crude[i] for i in trained], [size[i] for i in trained]),
                   "spearman_1minusR_vs_size": spearman([1 - R[i] for i in trained], [size[i] for i in trained]),
                   "n": len(trained)},  # fmt: skip
    }
    (rr.OUT / "crude_norm.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "known"} | {"known": {k: out["known"][k] for k in ("values", "orders", "margin", "lowest_reuse")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
