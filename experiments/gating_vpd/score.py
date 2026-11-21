"""P1.45 (E241): the band and the filed rules (results/predictions.json P1, P1c, P2, P2c, D1, D2) from the item files
present. No GPU, no torch:

    python experiments/gating_vpd/score.py        (writes results/summary.json)

A value of None is an item that is missing or unscored (KL(F||M) <= f on its windows): it enters no rule. The pure rules
are tested on their edge cases in tests/test_gating_vpd.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"
REUSE, NEW = ("x2", "x10", "moved", "rebuilt"), ("new_few", "new_thin")
SCORES = ("S", "S_c")
Vals = dict[str, float | None]


def verdict(gap: float, band: float, fail: str) -> str:
    return "holds" if gap > band else fail if gap <= 0 else "no call"


def order_rule(v: Vals, reuse: tuple[str, ...], new: tuple[str, ...], band: float) -> dict[str, Any]:
    """P1: min over the scored reuse items minus max over both new items; not read if a new item or every reuse item is None."""
    left = [i for i in reuse if v.get(i) is None]
    if any(v.get(i) is None for i in new) or len(left) == len(reuse):
        return {"read": False, "left_out": left}
    gap = min(x for i in reuse if (x := v.get(i)) is not None) - max(x for i in new if (x := v.get(i)) is not None)
    return {"read": True, "gap": gap, "band": band, "left_out": left, "verdict": verdict(gap, band, "rival b")}


def below_rule(reuse: list[float | None], rand: list[float | None], band: float) -> dict[str, Any]:
    """P2: min over the scored reuse items minus max over the scored random readings; not read if either list is empty."""
    r, x = [a for a in reuse if a is not None], [a for a in rand if a is not None]
    if not r or not x:
        return {"read": False, "n_reuse": len(r), "n_random": len(x)}
    gap = min(r) - max(x)
    return {"read": True, "gap": gap, "band": band, "n_reuse": len(r), "n_random": len(x), "verdict": verdict(gap, band, "fails")}


def band(halves: list[tuple[float | None, float | None]]) -> tuple[float, bool]:
    """max |half 1 - half 2| over the pairs where both are read; (0, True) when degenerate."""
    d = [abs(a - b) for a, b in halves if a is not None and b is not None]
    b = max(d, default=0.0)
    return b, b == 0


READ = {"S": "scored", "S_c": "scored_c"}  # S needs KL(F||M) > f; S_c, with f taken off, KL(F||M) > kl_floor


def value(it: dict[str, Any] | None, key: str) -> float | None:
    return it[key] if it is not None and it[READ[key]] else None


def pair_counts(index: dict[str, dict[str, list[str]]], items: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    """D1: per pair the mean score over each side's scored states; reuse above new among pairs with both sides read."""
    per: dict[str, Any] = {}
    for p, sides in index.items():
        m = {s: [x for n in sides[s] if (x := value(items.get(n), key)) is not None] for s in ("reuse", "new")}
        per[p] = {s: fmean(v) if v else None for s, v in m.items()} | {f"n_{s}": len(v) for s, v in m.items()}
    both = [p for p, r in per.items() if r["reuse"] is not None and r["new"] is not None]
    return {"per_pair": per, "n_read": len(both), "reuse_above": sum(per[p]["reuse"] > per[p]["new"] for p in both)}


def load_items() -> dict[str, dict[str, Any]]:
    its = (json.loads(p.read_text()) for p in (RESULTS / "items").glob("*.json") if not p.stem.endswith(".stopped"))
    return {it["item"]: it for it in its}


def summary(items: dict[str, dict[str, Any]], index: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    rand = {s: [n for n in items if n.startswith(f"random_{s}_")] for s in ("copy", "cue")}
    out: dict[str, Any] = {"experiment": "E241", "missing": [i for i in (*REUSE, *NEW) if i not in items]}
    for k in SCORES:  # the band: the six built items' halves, where the item is read on that score
        hv = [(items[n]["halves"][0][k], items[n]["halves"][1][k]) for n in (*REUSE, *NEW) if value(items.get(n), k) is not None]
        b, degenerate = band(hv)
        v = {i: value(items.get(i), k) for i in (*REUSE, *NEW)}
        tag = "" if k == "S" else "c"
        out[f"band_{k}"], out[f"band_{k}_degenerate"] = b, degenerate
        out[f"P1{tag}"] = order_rule(v, REUSE, NEW, b)
        out[f"P2{tag}"] = below_rule([v[i] for i in REUSE], [value(items[n], k) for n in rand["copy"]], b)
        out[f"D1_{k}"] = pair_counts(index, items, k)
    cols = ("scored", "scored_c", "S", "S00", "S_c")
    out["table"] = {n: {c: it[c] for c in cols} | {"kl_fm": it["means"]["kl_fm"], "f": it["means"]["f"], "refit_adds": None if it["S"] is None else it["S"] - it["S00"]}
                    for n, it in items.items() if not n.startswith("do_nothing")}  # fmt: skip
    return out


def main() -> int:
    idx_path = RESULTS / "pairs_index.json"
    out = summary(load_items(), json.loads(idx_path.read_text()) if idx_path.is_file() else {})
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    for k in ("band_S", "band_S_c", "P1", "P1c", "P2", "P2c"):
        print(k, json.dumps(out[k]))
    for k in SCORES:
        d = out[f"D1_{k}"]
        print(f"D1 {k}: reuse side above new in {d['reuse_above']} of {d['n_read']} pairs read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
