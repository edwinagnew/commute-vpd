"""E152 step 1 (cell P3.3): the planted sets and the filed rules, apart from the GPU job so both run on a CPU (the
dry read of the sets, the rule tests). Rules and thresholds: results/predictions_step1.json; step1.py reads them there.

Sets. At every down_proj matrix (the residual-stream writes; the MLP matrices are the ones the least-squares split
reaches whole, step 0), from W0's g statistics on the maths rows (job 56760, `active` = share of positions with g > 0.01):
  often  = the alive subcomponents with active > 0.5 (on at most maths positions);
  rarely = as many alive subcomponents at the same matrix with the lowest active (ties by mean g, then position).
The planted change rescales the write vectors of a set: u_c -> (1 + b) u_c, so dW_s = b U_P V_Pᵀ.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from torch import Tensor

PLANT_KIND = "down"
OFTEN_MIN = 0.5  # share of maths positions with g > 0.01 above which a subcomponent is on "at most" positions
RARELY_MAX = 0.01  # E45's gated-on share (qwen_reading.FRAC): a rarely-on member must sit under it
SIZES = ("rl", "mid", "sft")
NAN = float("nan")


def site_sets(active: Tensor, mean_g: Tensor) -> dict[str, Tensor]:
    """Positions into one matrix's alive list (active, mean_g already restricted to it): often and rarely, sorted."""
    a, g = active.double(), mean_g.double()
    often = (a > OFTEN_MIN).nonzero().flatten()
    order = g.argsort(stable=True)
    order = order[a[order].argsort(stable=True)]
    return {"often": often, "rarely": order[: len(often)].sort().values}


def planted_sets(maths: Mapping[str, Tensor], idx: Mapping[str, Tensor], sites: list[str]) -> dict[str, dict[str, Tensor]]:
    """kind -> site -> positions into that site's alive list, at the down_proj sites in `sites` (SITES order)."""
    out: dict[str, dict[str, Tensor]] = {"often": {}, "rarely": {}}
    for s in sites:
        if s.endswith(f"{PLANT_KIND}_proj"):
            i = idx[s].cpu()
            for k, v in site_sets(maths[f"{s}.active"][i], maths[f"{s}.mean_g"][i]).items():
                out[k][s] = v
    return out


# ----------------------------------------------------------------------------- the filed rules


def ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def share(part: float, whole: float) -> float:
    return part / whole if ok(part) and ok(whole) and whole > 0 else NAN


def recovery(planted: float, chance: float, n: int) -> float:
    """How far the planted share sits from chance towards 1: (P - C) / (1 - C); NaN for an empty set (n = 0, where
    both shares are 0 and would read as a smear), a missing share, or C >= 1."""
    return (planted - chance) / (1 - chance) if n > 0 and ok(planted) and ok(chance) and chance < 1 else NAN


def kind_call(r: float, P: Mapping[str, Any]) -> str:
    """'returns' (r >= returns_min), 'smears' (r < smears_max), 'partial' between, 'unreadable' (r NaN)."""
    if not ok(r):
        return "unreadable"
    r9 = round(r, 9)
    return "returns" if r9 >= P["returns_min"] else "smears" if r9 < P["smears_max"] else "partial"


def split_call(r_often: float, r_rarely: float, on_often: float, on_rarely: float, P: Mapping[str, Any]) -> dict[str, Any]:
    """Rows 1 and 3. 'works': both kinds return and the already-on share of the often kind exceeds the rarely kind's
    by at least apart_min (absolute); 'free': both kinds smear; otherwise 'no call'. Linear in the change, so one call
    holds at every size."""
    calls = {"often": kind_call(r_often, P), "rarely": kind_call(r_rarely, P)}
    gap = on_often - on_rarely if ok(on_often) and ok(on_rarely) else NAN
    apart = ok(gap) and round(gap, 9) >= P["apart_min"]
    if calls["often"] == calls["rarely"] == "returns" and apart:
        call = "works"
    elif calls["often"] == calls["rarely"] == "smears":
        call = "free"
    else:
        call = "no call"
    return {"kinds": calls, "on_gap": gap, "apart": apart, "call": call}


def separated(gap: float, gap95: list[float], spread: float, scored: bool) -> bool | None:
    """E132's rule for one pair of re-gating shares: |gap| > the seed spread and the 95% row-resample interval of the
    gap excludes 0. None (unreadable) when either state is unscored or unmatched, or a number is missing."""
    if not scored or not all(ok(x) for x in (gap, spread, *gap95)):
        return None
    return abs(gap) > spread and not gap95[0] <= 0 <= gap95[1]


def seen(sep: bool | None, matched: bool, statuses: tuple[str, ...]) -> bool | None:
    """The pair at one size for row 2: E132's rule where both shares are scored; False where a matched state is
    unscored (its denominator under E45's floor: the reading cannot see that size); None when a state is unmatched
    or a share is not finite."""
    if not matched or "nan" in statuses:
        return None
    return sep if all(s == "scored" for s in statuses) else False


def size_call(apart: Mapping[str, bool | None]) -> dict[str, Any]:
    """Row 2, on the re-gating share of the often/rarely pair at each size (SIZES, smallest first). The floor is the
    smallest size from which the pair separates at that size and every larger one. 'all sizes' (floor rl), 'too small
    below <floor>' (floor mid or sft), 'none' (not separated at sft), 'no call' (a size unreadable, or separated at a
    size but not at a larger one)."""
    v = [apart.get(s) for s in SIZES]
    if any(x is None for x in v):
        return {"floor": None, "call": "no call"}
    first = next((i for i, x in enumerate(v) if x), None)
    if first is None:
        return {"floor": None, "call": "none"}
    if not all(v[first:]):
        return {"floor": None, "call": "no call"}
    return {"floor": SIZES[first], "call": "all sizes" if first == 0 else f"too small below {SIZES[first]}"}


def blind_call(pairs: list[bool | None]) -> str:
    """Row 4 over every pair of the nine states: 'blind' when no readable pair separates and at least one is readable,
    'responds' when any pair separates, 'no call' when none is readable."""
    if not any(p is not None for p in pairs):
        return "no call"
    return "responds" if any(pairs) else "blind"
