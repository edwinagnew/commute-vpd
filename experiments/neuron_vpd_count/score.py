"""E117's score: E12's separation rule on panels A, B and C for both methods, and E53's 0.3 bar beside it on B and C.
Reads results/cells.json and results/predictions.json (its `separation_rule` and `e53_bar`); writes results/score.json
and prints every verdict. No model.

Usage (repo root):  ~/commute-agents/fleet/cpu-run python experiments/neuron_vpd_count/score.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"
PANELS, METHODS = ("A", "B", "C"), ("vpd", "neurons")


def readable(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def separation(existing: list[float | None], new: list[float | None]) -> dict[str, Any]:
    """E12's rule, existing predicted higher: 'separates' iff min(E) - max(N) > the larger within-class range (strict);
    'unreadable' with fewer than 2 readable values in a class (an unreadable value is dropped first)."""
    E, N = [float(x) for x in existing if x is not None and readable(x)], [float(x) for x in new if x is not None and readable(x)]
    if len(E) < 2 or len(N) < 2:
        return {"verdict": "unreadable", "existing": existing, "new": new}
    within = max(max(E) - min(E), max(N) - min(N))
    gap, swapped = min(E) - max(N), min(N) - max(E)
    verdict = "separates" if gap > within else "does not separate"
    return {"verdict": verdict, "direction": "reversed" if swapped > within else "", "gap": gap, "within": within,
            "existing": E, "new": N}


def e53_bar(low: list[float | None], ciphers: list[float | None], bar: float) -> dict[str, Any]:
    """min(lowercase) - max(both ciphers) >= bar; 'unreadable' when any value is."""
    if not all(readable(x) for x in low + ciphers):
        return {"verdict": "unreadable", "lowercase": low, "ciphers": ciphers}
    d = min(x for x in low if x is not None) - max(x for x in ciphers if x is not None)
    return {"verdict": "meets the bar" if d >= bar else "misses the bar", "difference": d, "bar": bar}


def value(cells: dict[str, Any], panel: str, method: str, name: str) -> float | None:
    """One state's (B, C) or one task's (A) value, None when unreadable (K_VPD = 0, or |G| under the floor)."""
    if panel == "A":
        j = cells["jaccard"][name]
        return j[method] if j["readable"] else None
    s = cells["rows"][name][panel][method]
    return s["set"] if s["readable"] else None


def score(cells: dict[str, Any], P: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for comp, c in P["separation_rule"]["comparisons"].items():
        for panel in PANELS:
            e, n = (c["A_existing"], c["A_new"]) if panel == "A" else (c["existing"], c["new"])
            for method in METHODS:
                out[f"{comp}/{panel}/{method}"] = separation([value(cells, panel, method, x) for x in e], [value(cells, panel, method, x) for x in n])
    bar = P["e53_bar"]
    low, ciphers = ["lowercase_s0", "lowercase_s1"], ["e53_cipher_s0", "e53_cipher_s1", "broad_cipher_s0", "broad_cipher_s1"]
    for panel in bar["panels"]:
        for method in METHODS:
            out[f"e53_bar/{panel}/{method}"] = e53_bar([value(cells, panel, method, x) for x in low], [value(cells, panel, method, x) for x in ciphers], bar["bar"])
    return out


def main() -> None:
    cells, P = json.loads((RESULTS / "cells.json").read_text()), json.loads((RESULTS / "predictions.json").read_text())
    assert cells.get("done"), "cells.json is not complete"
    res = score(cells, P)
    (RESULTS / "score.json").write_text(json.dumps({"cells_git_head": cells["git_head"], "verdicts": res}, indent=1, sort_keys=True))
    for k, v in res.items():
        nums = {q: round(v[q], 4) for q in ("gap", "within", "difference") if q in v}
        print(f"[score {k}] {v['verdict']} {v.get('direction', '')} {nums}")


if __name__ == "__main__":
    main()
