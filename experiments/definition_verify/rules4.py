"""E99 stage 4's rules, pure (no model): R-adm's two clauses, R-band's per-state reads with its pooling clause and E95's paired
test, the copy-against-lookup verdict through R-net, and the class map's verdict. Plan: reports/plans/2026-11-05_plan_e99-refine-
verify-definition.md ("Refinements adopted", stage 4); Fable: reports/reviews/2026-11-05_review_e99-fable-advice.md (2)-(4).
Thresholds and expected values: results/predictions_stage4.json. Comparisons are strict, as E90's: a tie is not above, and a nan
is never above, within, spared or admissible.

What each rule reads on the do-nothing state (W0 against itself, gain 0, share nan): verdict2 "unreadable: no gain"; own
"unreadable"; paired "unreadable"; reader "not read"; rnet_verdict and class_verdict "unreadable: no gain" per seed. So a
do-nothing state passes no bar.

Edge cases, each run before filing (~/commute-data/scratch/definition-lead/e99-stage4/edge4.py, its output edge4.out):
  merges           (0, 0), (5, 1), (1, 1) -> False; (2, 1) -> False (one value); (4, 2) -> True (a tie at 2 passes: "at least").
  reader           []; a failed match; a nan share or band -> "not read"; share == band -> "passes" (not above); one read
                   above among several matched -> "fails"; a negative share -> "passes".
  verdict2         no teeth -> "unreadable: no teeth" first; nan share -> "no gain"; nan control or band, or a failed match ->
                   "no call"; ties read within.
  redraw_spread    under two finite draws -> nan; all draws equal -> 0 up to rounding (2e-17 read).
  pooling          one state, or nan bands -> not pooled; spread > gap -> pooled to the largest; spread == gap -> not pooled.
  paired           a nan share or no finite draw -> "unreadable" (E95's `read_rules.paired`, unchanged).
  rnet_verdict     a forced arm off 1, a forbidden arm not within, or no arm -> "unreadable: R-net's controls failed";
                   unequal seed counts -> ValueError (zip strict).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import rules as R

from commute.copying import finite
from commute.jobs import load_experiment

RR = load_experiment("token_identity", "read_rules")  # E95: paired, over_seeds
R90 = load_experiment("new_vs_reuse", "rules")  # E90: the verdict strings
RA = RR.RA
ABOVE, WITHIN, MIXED, UNREADABLE = RA.ABOVE, RA.WITHIN, RA.MIXED, RA.UNREADABLE
ADMISSIBLE, NO_C = "admissible", "no C listed"
NO_ADMISSIBLE = "no admissible C at this site: the removal deletes the input"
B_UNREAD = "unreadable: R-adm (b) not read at this site, no token-only reader downstream of it"
PASSES, FAILS, NOT_READ = "passes", "fails", "not read"
CONTROLS_FAILED = "unreadable: R-net's controls on this part failed"


# ---- R-adm ----


def merges(n_inputs: int, n_values: int, min_per_value: float) -> bool:
    """R-adm (a): f_C merges inputs on X_C, more than one value and at least `min_per_value` inputs per value on average. An input
    is what f_C is a function of: the token for token identity and the embedding class, the context window for W0's guess."""
    return n_values > 1 and n_inputs / n_values >= min_per_value


def reader(reads: Sequence[Mapping[str, Any]]) -> str:
    """R-adm (b) from the token-only reader's reads downstream of the site, each {"share", "band", "status"} (status: the write's
    gain match, "ok" when matched). No read, a failed match or a nan voids the clause ("not read", Fable: say so rather than pass
    by default); else "fails" when any read is above its own band, "passes" when every read is within it."""
    if not reads or any(r.get("status") != "ok" or not (finite(r["share"]) and finite(r["band"])) for r in reads):
        return NOT_READ
    return FAILS if any(r["share"] > r["band"] for r in reads) else PASSES


def admissibility(e90_label: str, merges_ok: bool, reader_status: str) -> str:
    """E90's admissibility (decisions.md row 116) first, then R-adm (a), then (b). "admissible" or the row's reason."""
    if e90_label != ADMISSIBLE:
        return NO_C
    if not merges_ok or reader_status == FAILS:
        return NO_ADMISSIBLE
    return ADMISSIBLE if reader_status == PASSES else B_UNREAD


# ---- R-band ----


def own(share: float, band: float) -> str:
    """One read against its own band: E90 Amendment 2's `status`, strict; a nan is unreadable."""
    return RA.status([share], band, True)


def verdict2(teeth: bool, s: float, s_band: float, c: float, c_band: float, control_ok: bool = True) -> str:
    """E90's 2 x 2 (`rules.verdict`) with the state and its spared control each against its own band (R-band)."""
    if not teeth:
        return R90.NO_TEETH
    if not finite(s):
        return R90.NO_GAIN
    if not (finite(c) and finite(s_band) and finite(c_band)) or not control_ok:
        return R90.NO_CALL
    st, ct = s > s_band, c > c_band
    return R90.NOT_SPECIFIC if st and ct else R90.REUSES if st else R90.NOT_THROUGH if ct else R90.NO_REACH


def redraw_spread(draws: Sequence[float], n_sets: int, seed: int) -> float:
    """The band's spread under redraws: the standard deviation of `band_of` over n_sets sets of as many values, drawn from a normal
    fit to the finite draws, as the E90 check simulated (its 20,000 sets). nan under two finite draws."""
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return math.nan
    sims = np.random.default_rng(seed).normal(a.mean(), a.std(ddof=1), size=(n_sets, a.size))
    return float(np.percentile(np.abs(sims), 95, axis=1).std())


def pooling(rows: Mapping[str, Mapping[str, Any]], n_sets: int, seed: int) -> dict[str, Any]:
    """R-band's clause over the states read at one C and block (rows hold "band" and "random"): pool to the largest band only if the
    largest redraw spread exceeds the gap between the states' bands, largest minus smallest. Under two finite bands: not pooled."""
    bands = {n: float(r["band"]) for n, r in rows.items() if finite(r["band"])}
    if len(bands) < 2:
        return {"pooled": False, "spread": math.nan, "gap": math.nan, "largest": max(bands.values(), default=math.nan)}
    spread = max(redraw_spread(rows[n]["random"], n_sets, seed) for n in bands)
    gap = max(bands.values()) - min(bands.values())
    return {"pooled": bool(finite(spread) and spread > gap), "spread": spread, "gap": gap, "largest": max(bands.values())}


def band_for(band: float, pool: Mapping[str, Any]) -> float:
    return pool["largest"] if pool["pooled"] else band


def paired(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """E95's filed paired test, unchanged: rows {"A_lo", "random"} per seed, the same draws in the same order."""
    return RR.paired(a, b)


def over_seeds(per_seed: Sequence[str]) -> dict[str, Any]:
    return R90.over_seeds(per_seed)


# ---- (a) the copy-against-lookup verdict through R-net ----


def rnet_verdict(states: Sequence[tuple[float, float]], controls: Sequence[tuple[float, float]], forced: Sequence[str], forbidden: Sequence[str]) -> dict[str, Any]:
    """States and their spared controls as (net share, own band) per seed, paired by seed; the forced arms' readings must all be
    "net 1" (the positive control) and the forbidden arms' all "within band" (the negative), else the part reads nothing."""
    pos, neg = bool(forced) and all(f == R.NET_ONE for f in forced), bool(forbidden) and all(f == R.WITHIN for f in forbidden)
    per = [verdict2(True, s, sb, c, cb) for (s, sb), (c, cb) in zip(states, controls, strict=True)]
    v = over_seeds(per) if pos and neg else {"verdict": CONTROLS_FAILED, "per_seed": per}
    return {**v, "positive_control": pos, "negative_control": neg}


# ---- (d) the class map ----


def misread(share: float, spared_max: float) -> bool:
    """The hand-write control's stop: a write upstream of the removal losing more than `spared_max` of its gain."""
    return finite(share) and share > spared_max


def class_verdict(adm: str, teeth: bool, states: Sequence[tuple[float, float]], writes: Sequence[tuple[float, float, bool]]) -> dict[str, Any]:
    """Admissibility first; then E90's 2 x 2 per seed against own bands, the control the same seed's hand write upstream."""
    if adm != ADMISSIBLE:
        return {"verdict": adm, "per_seed": []}
    return over_seeds([verdict2(teeth, s, sb, c, cb, ok) for (s, sb), (c, cb, ok) in zip(states, writes, strict=True)])
