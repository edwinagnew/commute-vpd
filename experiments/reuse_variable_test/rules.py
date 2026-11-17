"""E137's filed rules, pure functions over numbers read in the run. Every threshold comes from results/predictions.json
("thresholds"), passed in as `th`, so the run and the filed text cannot disagree. No model, no torch."""

from __future__ import annotations

import math
from typing import Any

PAIRS = ("Always on", "Cue", "XOR", "Table")
NO_READING = "no reading"
EPS = 1e-12  # 1.0 - 0.9 is 0.09999... in floats: a tie at the margin must read as a tie


def _bad(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def identity_ok(max_d: float, floor: float) -> bool:
    """M' with a piece at its start against M: outside the float floor stops the job. Zero against a zero floor passes."""
    return not _bad(max_d) and not _bad(floor) and max_d <= floor


def reading(gap: float, th: dict[str, float]) -> bool:
    """A pair is read only if its mean gap reaches the floor (0.02 is `th["floor_gap"]`)."""
    return not _bad(gap) and gap >= th["floor_gap"]


def tol_share(tol_tv: float, gap: float) -> float:
    """A share (G, H or reversion) is 1 minus a mean distance over gap, so the tolerance on the share is the tolerance on
    the distance over gap. inf for a zero gap, which `reading` has already refused."""
    return tol_tv / gap if gap > 0 else math.inf


def tol_mismatch(tol_h: float, var: float) -> float:
    """The mismatch is a mean squared distance over the variance of v_F. Zero variance: 0.0, so only an exact 0 passes
    (the wording's convention sets the mismatch to 0 or 1 there)."""
    return tol_h * tol_h / var if var > 0 else 0.0


def within(x: float | None, want: float, tol: float) -> bool:
    return not _bad(x) and abs(float(x) - want) <= tol  # type: ignore[arg-type]


def search_bar(found: float | None, witness: float | None, th: dict[str, float]) -> bool:
    """The search met the bar if the found value is at least th["search_bar"] of the witness value. A missing number misses."""
    return not _bad(found) and not _bad(witness) and float(found) >= th["search_bar"] * float(witness)  # type: ignore[arg-type]


def built_possible(accs: list[float | None], th: dict[str, float]) -> bool:
    """XOR can read built only if no affine read of it from z_M exceeds the bar held out, at any site's z_M. An empty
    list or a missing number cannot show that, so it is False."""
    return bool(accs) and not any(_bad(a) for a in accs) and max(float(a) for a in accs if a is not None) <= th["capacity_bar"]


def card_rule(g: dict[str, list[float | None] | None], th: dict[str, float]) -> str:
    """The card's call at k = 4: `g` maps a pair to its three read G's at its best site. "A" if Always on's mean is at
    least the margin above each of Cue, XOR, Table; "B" if each is within the margin of Always on; "mixed" otherwise;
    `NO_READING` if a pair has no reading or a start is missing. A is tried first, so a tie of exactly the margin reads A."""
    if any(g.get(p) is None for p in PAIRS):
        return NO_READING
    means: dict[str, float] = {}
    for p in PAIRS:
        v = g[p]
        assert v is not None
        if len(v) == 0 or any(_bad(x) for x in v):
            return NO_READING
        means[p] = sum(float(x) for x in v if x is not None) / len(v)
    a, m = means["Always on"], th["card_margin"]
    rest = [means[p] for p in PAIRS[1:]]
    if all(a - x >= m - EPS for x in rest):
        return "A"
    if all(abs(a - x) <= m + EPS for x in rest):
        return "B"
    return "mixed"


def best_site(by_site: dict[str, list[float]]) -> tuple[str | None, list[float] | None]:
    """The site with the highest read G over the starts, ties to the earlier site in the dict's order. Nothing read: (None, None)."""
    if not by_site:
        return None, None
    site = max(by_site, key=lambda s: max(by_site[s]))
    return site, by_site[site]


def acted_on_count(mean_ds: list[float], tol_tv: float) -> int:
    """How many directions moved the output by more than the float floor (zero directions: 0, an empty count)."""
    return sum(1 for d in mean_ds if d > tol_tv)


def rule_summary(filed: dict[str, Any]) -> dict[str, Any]:
    """The constants the run reports beside each reading, verbatim from the file."""
    return {"by_construction": filed["by_construction"], "thresholds": filed["thresholds"]}
