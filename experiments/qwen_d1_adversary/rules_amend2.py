"""E80 Amendment 2's filed rules (results/predictions_amend2.json), pure functions of the read numbers, so that edge cases can be run on them. The Qwen column of the slide's
scorecard: Faithful from the adversary on W, Behaviour kept from the held-out gold log-loss with no masks. Bars are compared after rounding to 9 places, as rules.py does, so a
number exactly on a bar is on it. A NaN, a missing start or a zero W0 reads "unread"."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import rules
import rules_amend1

NAN = math.nan
UNREAD = "unread"
PASS, FAIL, BETWEEN = "pass", "fail", "between"
INSIDE_SPREAD, SPREAD_UNREAD = "over the fail bar, inside the start spread", "over the fail bar, spread unread"
ROWS = ("N", "LS", "ADJ", "A")
PASS_AT, FAIL_AT = rules.PASS_AT, rules.FAIL_AT
E80_SPREAD = 0.16  # W0 on E80's M rows, 3.999 against 3.443: quoted beside this job's spread, never ruled on
CANT_TELL = "we can't yet tell on Qwen"
KEPT_TOL = 1e-3  # do-nothing and least squares against 1.00: scorecard_ls/results/predictions.json's "s 1 to 1e-3". N_clean's measured KL through the two paths is 1.2e-9
W0_GOLD, W0_TOL = 2.4952089256294077, 1e-3  # qwen_sft/results/A0.json gold_loss; E71's w0_tol. These rows' reading at W0, so a wrong row set stops the job
FT_GAIN, FT_TOL = 1.810801895590828, 0.05  # qwen_sft/results/F.json saved.10.gold_drop, E40's own fp32 weights; here the saved bf16 regen file. Chosen here: rounding reads well inside, another file or rows far outside
E68_S, E68_TOL = {"energy": 1440.092819832963, "LS": 0.3637434871295139, "ADJ": 0.0004657108503729808}, 0.01  # qwen_pass_test/results/u1/S.json: the same states, on an L40


def mean_r(p: rules_amend1.Pair) -> float:
    """Mean of (r_0, r_1); NaN if either is unread."""
    return NAN if any(math.isnan(x) for x in p) else sum(p) / len(p)


def verdict(m: float) -> str:
    """Both bars inclusive; "between" otherwise."""
    if math.isnan(m):
        return UNREAD
    return PASS if rules.r9(m) <= PASS_AT else FAIL if rules.r9(m) >= FAIL_AT else BETWEEN


def spread(a: float, b: float) -> float:
    """W0's two starts, |a - b| / the smaller. NaN if one is unread or not above zero, or if they are equal: two seeds reading the same number means no spread was measured."""
    if math.isnan(a) or math.isnan(b) or min(a, b) <= 0 or rules.r9(a) == rules.r9(b):
        return NAN
    return abs(a - b) / min(a, b)


def described(m: float, s: float) -> str:
    """What a row may be called. Only "fail" says a method fails: over the bar and its gap, mean r - 1, strictly above W0's spread `s` (a gap on the spread has not beaten it)."""
    v = verdict(m)
    if v != FAIL:
        return v
    if math.isnan(s):
        return SPREAD_UNREAD
    return FAIL if rules.r9(m - 1) > rules.r9(s) else INSIDE_SPREAD


def reading(n: str) -> str:
    """The table's reading from do-nothing's row, `described`. The column can fail a method only if it fails do-nothing."""
    if n == UNREAD:
        return UNREAD
    if n == FAIL:
        return "the column fails do-nothing: each row reads as its verdict"
    if n == BETWEEN:
        return "do-nothing reads between the bars: the column has not failed it, rows reported as read"
    return CANT_TELL  # do-nothing passes, or is over the bar only inside the spread


def kept(x: float, w0: float, ft: float) -> float:
    """Share of the gain kept, from three mean log-losses: (w0 - x) / (w0 - ft). 0 is the unchanged model, 1 the fine-tune. NaN if one is unread or the fine-tune gained nothing."""
    if any(math.isnan(v) for v in (x, w0, ft)) or w0 - ft <= 0:
        return NAN
    return (w0 - x) / (w0 - ft)


def near(x: float, want: float, tol: float) -> bool:
    """Within tol, its end included; NaN is not."""
    return rules.r9(abs(x - want)) <= tol


def near_rel(x: float, want: float, tol: float) -> bool:
    return near(x / want, 1.0, tol)


def lean(cell: Any, holds: bool) -> bool | None:
    return None if cell == UNREAD or (isinstance(cell, float) and math.isnan(cell)) else holds


def score(starts: Mapping[str, Any], gold: Mapping[str, float], checks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`starts`: key -> {"kl": x}, this job's LS and ADJ and Amendment 1's W0, N, A. `gold`: read -> mean log-loss. "unread" for what has not been read."""
    r = {x: rules_amend1.pair(starts, x) for x in ROWS}
    m = {x: mean_r(p) for x, p in r.items()}
    s = spread(rules_amend1.kl(starts, "W0_W_s0"), rules_amend1.kl(starts, "W0_W_s1"))
    row = {x: described(m[x], s) for x in ROWS}
    w0, ft = gold.get("W0", NAN), gold.get("FT", NAN)
    share = {x: kept(gold.get(x, NAN), w0, ft) for x in (*ROWS, "W0_end")}
    return {"r": {x: list(p) for x, p in r.items()}, "mean": m, "verdict": {x: verdict(m[x]) for x in ROWS}, "gap": {x: m[x] - 1 for x in ROWS},
            "W0_spread": s, "E80_spread_quoted": E80_SPREAD, "described": row, "reading": reading(row["N"]),
            "kept": share, "gain": w0 - ft, "gold": dict(gold), "checks": dict(checks or {}),
            "last5": {k: v["last5"] for k, v in starts.items() if isinstance(v, Mapping) and "last5" in v},
            "leans": {"N_passes": lean(row["N"], row["N"] == PASS), "LS_passes": lean(row["LS"], row["LS"] == PASS),
                      "ADJ_passes": lean(row["ADJ"], row["ADJ"] == PASS), "A_passes": lean(row["A"], row["A"] == PASS),
                      "ADJ_under_half": lean(share["ADJ"], share["ADJ"] < 0.5), "A_over_half": lean(share["A"], share["A"] > 0.5)}}
