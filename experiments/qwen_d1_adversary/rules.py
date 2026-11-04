"""E80's filed rules (results/predictions.json), pure functions of the read numbers, so that edge cases can be run on them. All on M; r = a state's seed 0 / W0's seed 0.
Bars are compared after rounding to 9 places, so a number exactly on a bar is on it. A NaN, a missing number or a zero W0 reads "unread"."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

NAN = math.nan
W_128 = 1.7016  # the 128-row job, seed 0
STAGE1_LOW, STAGE1_HIGH = 1.36, 2.04  # a fifth either way of 1.70
BAND_MIN = 0.19  # the spread measured over three starts on W
PASS_AT, FAIL_AT = 1.40, 1.61  # E71's bars
N_CLEAN_UNDER = 1e-4
I_CLEAN_MAX = 0.05
ROWS_TOTAL, ROW_FULL, M_FROM, M_ROWS, ROW_LEN = 128, 512, 64, 64, 256


def r9(x: float) -> float:
    return round(x, 9)


def m_rows(maths: Sequence[Sequence[int]], used: Sequence[Sequence[int]]) -> list[list[int]]:
    """M = maths rows 64-127 cut to 256 tokens; asserted to be 64 rows of 256 and to share no row with `used`, E76's rows cut alike."""
    assert len(maths) == ROWS_TOTAL and all(len(r) == ROW_FULL for r in maths), "maths_rows.json: 128 rows of 512"
    M = [list(r[:ROW_LEN]) for r in maths[M_FROM:]]
    assert len(M) == M_ROWS and all(len(r) == ROW_LEN for r in M), "M: 64 rows of 256"
    assert {tuple(r) for r in M}.isdisjoint(tuple(r[:ROW_LEN]) for r in used), "M shares a row with E76's rows"
    return M


def stage1_ok(kl: float) -> bool:
    """Inside the window, its ends included; NaN is outside."""
    return STAGE1_LOW <= r9(kl) <= STAGE1_HIGH


def first_step_ok(kl: float) -> bool:
    return math.isfinite(kl)


def n_clean_ok(kl: float) -> bool:
    return r9(kl) < N_CLEAN_UNDER


def i_clean_ok(kl: float) -> bool:
    """The stop is "above": exactly on the bar goes on."""
    return r9(kl) <= I_CLEAN_MAX


def band(a: float, b: float) -> float:
    """(larger - smaller) / smaller, at least BAND_MIN; NaN when a start is unread or not above zero."""
    if math.isnan(a) or math.isnan(b) or min(a, b) <= 0:
        return NAN
    return max((max(a, b) - min(a, b)) / min(a, b), BAND_MIN)


def ratio(state: float, w0: float) -> float:
    return state / w0 if w0 > 0 and not math.isnan(state) else NAN


def do_nothing(r: float, b: float) -> str:
    """Strictly over 1 + b fails it; exactly on the bar is within the band."""
    if math.isnan(r) or math.isnan(b):
        return "unread"
    return "fails do-nothing" if r9(r) > r9(1 + b) else "within the band"


def bars(r: float) -> str:
    """Closed on both sides, as filed."""
    if math.isnan(r):
        return "unread"
    return "passes" if r9(r) <= PASS_AT else "fails" if r9(r) >= FAIL_AT else "no call"


def outcome(dn: str, inspan: str) -> str:
    if dn == "within the band":
        return "the adversary cannot fail do-nothing here"
    if dn == "unread" or inspan == "unread":
        return "unread"
    return "D1 has a test and a first candidate" if inspan == "passes" else "a test, no candidate"


def num(d: Mapping[str, Any], *keys: str) -> float:
    """The number at `keys`, NaN if any key is missing or the value is None."""
    x: Any = d
    for k in keys:
        if not isinstance(x, Mapping) or k not in x:
            return NAN
        x = x[k]
    return NAN if x is None else float(x)


def score(out: Mapping[str, Any]) -> dict[str, Any]:
    """From what has been read, "unread" for what has not. W, A, the own-mask reads and the weight errors are reported as read."""

    def kl(key: str) -> float:
        return num(out, "starts", key, "kl")

    w0 = kl("W0_M_s0")
    b = band(w0, kl("W0_M_s1"))
    r = {s: ratio(kl(f"{s}_M_s0"), w0) for s in ("N", "I", "A")}
    dn, ins = do_nothing(r["N"], b), bars(r["I"])
    return {"b": b, "r": r, "do_nothing": dn, "inspan": ins, "A": bars(r["A"]), "outcome": outcome(dn, ins),
            "W_as_read": {"W0": kl("W0_W_s0"), "N": kl("N_W_s0"), "r_N": ratio(kl("N_W_s0"), kl("W0_W_s0"))},
            "stage1_in_window": None if math.isnan(kl("W0_W_s0")) else stage1_ok(kl("W0_W_s0")),
            "own": out.get("own", {}), "weights": out.get("weights", {}), "checks": out.get("checks", {}),
            "leans": {"within_the_band": None if dn == "unread" else dn == "within the band", "inspan_passes": None if ins == "unread" else ins == "passes"}}
