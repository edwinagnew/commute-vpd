"""E80 Amendment 1's filed rules (results/predictions_amend1.json), pure functions of the read numbers, so that edge cases can be run on them.
All on W, the FineWeb rows. r_s(X) = X_s / W0_s, the final KL of seed s. Bars are compared after rounding to 9 places, as rules.py does, so a number exactly
on a bar is on it. A NaN, a missing start or a zero W0 reads "unread"."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import rules

NAN = math.nan
UNREAD = "unread"
BAND_MIN = 0.05  # the smallest seed-paired spread worth naming; a floor only, b itself is measured
SEEDS = (0, 1)
r9 = rules.r9
Pair = tuple[float, float]


def kl(starts: Mapping[str, Any], key: str) -> float:
    """The final KL of start `key` in a map key -> {"kl": x}; NaN if missing or None."""
    return rules.num(starts, key, "kl")


def pair(starts: Mapping[str, Any], state: str) -> Pair:
    """(r_0, r_1) of `state`: its seed-s final KL over W0's seed-s, NaN where either is unread or W0 is not above zero."""
    r0, r1 = (rules.ratio(kl(starts, f"{state}_W_s{s}"), kl(starts, f"W0_W_s{s}")) for s in SEEDS)
    return r0, r1


def spread(p: Pair) -> float:
    """|r_0 - r_1| / the smaller; NaN if either is unread or the smaller is not above zero."""
    if any(math.isnan(x) for x in p) or min(p) <= 0:
        return NAN
    return abs(p[0] - p[1]) / min(p)


def band(n: Pair, i: Pair) -> float:
    """The larger of N's and I's seed spread, at least BAND_MIN; NaN if either spread is unread."""
    s = (spread(n), spread(i))
    return NAN if any(math.isnan(x) for x in s) else max(*s, BAND_MIN)


def reproduces(r1_n: float) -> str:
    """r_1(N) > 1, strictly; otherwise the job stops after N seed 1."""
    if math.isnan(r1_n):
        return UNREAD
    return "reproduces" if r9(r1_n) > 1 else "does not reproduce"


def gap(n: Pair, x: Pair) -> float:
    """Mean over seeds of r_s(N) / r_s(X); NaN if any is unread or an r_s(X) is not above zero."""
    if any(math.isnan(v) for v in (*n, *x)) or min(x) <= 0:
        return NAN
    return sum(a / b for a, b in zip(n, x, strict=True)) / len(SEEDS)


def separated(n: Pair, x: Pair, b: float) -> str:
    """r_s(N) > r_s(X) on both seeds, strictly, and gap(X) > 1 + b, strictly. N as its own candidate has gap 1, so never separated."""
    g = gap(n, x)
    if math.isnan(g) or math.isnan(b):
        return UNREAD
    return "separated" if all(r9(a) > r9(c) for a, c in zip(n, x, strict=True)) and r9(g) > r9(1 + b) else "not separated"


def outcome(rep: str, i: str, a: str) -> str:
    """rep from `reproduces`, i and a from `separated`. "Not separated" for I decides without A, as a miss of the rise decides without I."""
    if rep == UNREAD:
        return UNREAD
    if rep != "reproduces":
        return "do-nothing's rise does not reproduce on W"
    if i == UNREAD:
        return UNREAD
    if i == "not separated":
        return "cannot tell in-span from do-nothing on W"
    if a == UNREAD:
        return UNREAD
    return "the adversary follows placement here" if a == "not separated" else "separates moved components from unmoved, not good from bad placement"


def score(starts: Mapping[str, Any], checks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`starts`: key -> {"kl": x}, E80's W0_W_s0 and N_W_s0 among them. "unread" for what has not been read."""
    r = {s: pair(starts, s) for s in ("N", "I", "A")}
    b = band(r["N"], r["I"])
    rep, i, a = reproduces(r["N"][1]), separated(r["N"], r["I"], b), separated(r["N"], r["A"], b)
    return {"r": {s: list(p) for s, p in r.items()}, "b": b, "gap": {"I": gap(r["N"], r["I"]), "A": gap(r["N"], r["A"])},
            "reproduces": rep, "separated": {"N": separated(r["N"], r["N"], b), "I": i, "A": a}, "outcome": outcome(rep, i, a),
            "last5": {k: v["last5"] for k, v in starts.items() if isinstance(v, Mapping) and "last5" in v}, "checks": dict(checks or {}),
            "leans": {"reproduces": None if rep == UNREAD else rep == "reproduces", "I_not_separated": None if i == UNREAD else i == "not separated"}}
