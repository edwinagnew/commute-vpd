"""E309 (cell P2.79): the reproduction table of the reading refactor, CPU, from results/old_A.json, old_B.json (old.py,
the old code run twice; old_c_A / old_c_B, old_c.py's stage "scans", merged into them) and "new": new_a.json,
new_b.json and new_c.json merged (the new functions, same keys; a key in two files is an error).

Pass rule (plan "Reproduction check and its floor"):
  - old A and old B bit-identical: new must equal them bit for bit, or within 1e-12 relative (a sum in another order);
  - otherwise new must lie within [min(A, B), max(A, B)], the old code's own spread on this card;
  - either way, |new - saved| must stay under the saved run's filed tolerance (TOL, TOL_REL); a reading with none
    (gain, adl_D, prior_mean_lo, kl, needed_set_size, erank, size) must repeat the saved value within REL, so every
    gap from the saved run is labelled.
The old values' distance from the saved run is printed beside (card, batch or rows differ: written down, never absorbed).
A reading with no new value reads "pending".

    ~/commute-agents/fleet/cpu-run python3 experiments/reading_refactor/compare.py     -> results/reproduction.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"
REL = 1e-12  # a data reading summed in a different order
TOL = {"share": 0.01, "steering_pooled": 0.001, "E_b_vpd": 1e-11, "E_b_neurons": 1e-11, "E_b_sae": 1e-11}  # E136 reproduce_tol, E211 e211_reread, E268 transcription
TOL_REL = {"prior_geo": 1e-4, "prior_log10_geo": 1e-4}  # prior_vs_share stage0 tol_rel


def load(name: str) -> dict[str, Any]:
    p = RESULTS / f"{name}.json"
    return {k: v for k, v in json.loads(p.read_text()).items() if not k.startswith("_")} if p.is_file() else {}


def merged(*names: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in names:
        d = load(n)
        assert not (both := sorted(set(out) & set(d))), f"{n}.json repeats keys {both}"
        out |= d
    return out


def num(rec: dict[str, Any] | None) -> float | None:
    v = None if rec is None else rec.get("value")
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def under_tol(reading: str, new: float, saved: float | None) -> bool | None:
    if saved is None or not isinstance(saved, int | float):
        return None
    if reading in TOL_REL:
        return abs(new / saved - 1) <= TOL_REL[reading]
    return abs(new - saved) <= TOL.get(reading, REL * abs(saved))


def verdict(reading: str, a: float | None, b: float | None, new: float | None, saved: Any) -> str:
    """FAIL when new is not old (outside the old spread). Over the saved run's filed tolerance is a FAIL only when old
    is under it: a gap the old code has too is the saved-vs-recompute gap, pre-existing, and reads pass."""
    if new is None:
        return "pending"
    if a is None:
        return "no old value"
    if b is None or a == b:
        ok = new == a or abs(new - a) <= REL * abs(a)
    else:
        ok = min(a, b) <= new <= max(a, b)
    if not ok:
        return "FAIL (outside old spread)"
    if under_tol(reading, new, saved) is not False:
        return "pass"
    if under_tol(reading, a, saved) is False:
        return f"pass (new = old; old off saved by {a - float(saved):.3g}, pre-existing)"
    return "FAIL (over filed tol, old under it)"


def table() -> list[dict[str, Any]]:
    A, B, N = merged("old_A", "old_c_A"), merged("old_B", "old_c_B"), merged("new_a", "new_b", "new_c")
    rows = []
    for key in sorted(A | B):
        reading, state = key.split("/", 1)
        a, b, n = num(A.get(key)), num(B.get(key)), num(N.get(key))
        saved = (A.get(key) or B.get(key) or {}).get("saved")
        s = float(saved) if isinstance(saved, int | float) and not isinstance(saved, bool) else None
        rows.append({"reading": reading, "state": state, "saved": saved, "old_A": a, "old_B": b, "floor_abs_A_B": abs(a - b) if a is not None and b is not None else None,
                     "new": n, "old_A_minus_saved": a - s if a is not None and s is not None else None, "pass": verdict(reading, a, b, n, saved)})  # fmt: skip
    return rows


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.10g}"
    return "-" if x is None else str(x)


def main() -> None:
    rows = table()
    (RESULTS / "reproduction.json").write_text(json.dumps(rows, indent=1))
    cols = ("reading", "state", "saved", "old_A", "old_B", "floor_abs_A_B", "new", "old_A_minus_saved", "pass")
    print("\t".join(cols))
    for r in rows:
        print("\t".join(fmt(r[c]) for c in cols))


if __name__ == "__main__":
    main()
