"""E214's substitute reading (user, 2026-11-19, plan amendment 5): the two hand-built reuse edits were not built, so the
pass rule's formula is read with E195's reuse items (x2, x10, moved, rebuilt) as the reuse side. This is a description,
not the filed rule R1. Per grade and level: min R over the reuse side minus max R over a new side, for two new sides,
the two built new edits and the six grafts; f > 0 for every reuse item; and the band as score.py computes it (band_factor
x the largest spread over sd of R across orders, |R(half 1) - R(half 2)|, tie range), here over the two new edits only,
since the filed band's four edits are not all read. Writes results/substitute.json. No GPU:

    python experiments/kstar_levels/substitute.py
"""

from __future__ import annotations

import json
import statistics
from typing import Any

from commute.jobs import load_experiment

S = load_experiment("kstar_levels", "score")
REUSE = ("x2", "x10", "moved", "rebuilt")
NEW = {"edits": ("new_few", "new_thin"), "grafts": ("e201", "last_lowest", "last_rank5", "last_rank9", "last_rank13", "neurons")}
LEVELS = ("head_mlp", "mlp_attn", "block")


def spreads(d: Any, item: str, L: str, g: str) -> dict[str, float | None]:
    m = [r[f"R_{g}"] for r in S.recs(d, item, L)]
    halves = all((item, L) in d and v in d[(item, L)]["searches"] for v in ("half1", "half2"))
    return {"order_sd": statistics.stdev(m) if len(m) > 1 else 0.0,
            "split_half": abs(S.R(d, item, L, g, "half1") - S.R(d, item, L, g, "half2")) if halves else None,
            "tie_range": S.recs(d, item, L)[0].get(f"R_{g}_tie_range", 0.0)}  # fmt: skip


def level(d: Any, L: str, g: str, k: float) -> dict[str, Any]:
    R = {i: S.R(d, i, L, g) for i in (*REUSE, *NEW["edits"], *NEW["grafts"])}
    sp = {i: spreads(d, i, L, g) for i in (*REUSE, *NEW["edits"])}
    band = k * max(v for i in NEW["edits"] for v in sp[i].values() if v is not None)
    out: dict[str, Any] = {"R": R, "f": {i: statistics.mean(r["f"] for r in S.recs(d, i, L)) for i in REUSE}, "spreads": sp, "band_new_edits": band}
    out["f_positive"] = all(v > 0 for v in out["f"].values())
    out["min_reuse"] = min(R[i] for i in REUSE)
    for side, items in NEW.items():
        out[f"max_{side}"] = max(R[i] for i in items)
        out[f"gap_{side}"] = out["min_reuse"] - out[f"max_{side}"]
        out[f"above_band_{side}"] = out[f"gap_{side}"] > band and out["f_positive"]
    return out


def main() -> None:
    pred = json.loads((S.RESULTS / "predictions.json").read_text())
    d, k = S.load(), pred["thresholds"]["band_factor"]
    out = {"experiment": pred["experiment"], "main_grade": pred["main_grade"], "reading": "description, not the filed R1",
           "reuse_side": REUSE, "new_sides": NEW} | {f"grade_{g}": {L: level(d, L, g, k) for L in LEVELS} for g in ("read", "all")}  # fmt: skip
    (S.RESULTS / "substitute.json").write_text(json.dumps(out, indent=1))
    for g in ("read", "all"):
        for L, v in out[f"grade_{g}"].items():
            print(f"{g} {L}: min reuse {v['min_reuse']:.4f}; edits gap {v['gap_edits']:+.4f}, grafts gap {v['gap_grafts']:+.4f}; band {v['band_new_edits']:.4f}; f>0 {v['f_positive']}")


if __name__ == "__main__":
    main()
