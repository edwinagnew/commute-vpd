"""E99's verdict rules, pure (no model): the net share, the per-state band, the expected readings of the forced and
forbidden fine-tunes, the matched gain and the stage-0 checks. Plan: reports/plans/2026-11-05_plan_e99-refine-verify-definition.md
("Refinements adopted", "Stages"); thresholds and references: results/predictions.json.

Net share (R-net): (the state's fall with the part at W0's means - W0's own fall with the part off) / the state's gain G,
unit "lo". It is the share of the update's increment that vanishes with the part, read in both models.
Band (R-band): per state, the 95th percentile of |net share| over the same state's random head pairs drawn outside the
part (`removal.band_of`, E90's band). A part reads "through the part" above the band, "not through" within it.

Expected value of each rule (W0 = the zero update; native = E62's copy step-30 / E53's lowercase step-300 states; numbers
from E97's stage1.json / stage2.json, copied into predictions.json "reference"):
  net_share        W0: nan (G = 0, "no gain").  native copy s0, s1: 0.72, 0.73.  native lowercase s0, s1: -0.18, -0.16.
                   native lookup at the copy pair: 0.014, 0.003.  forced: 1 (exact by construction: outside the pair's
                   q, k, v rows nothing moves and o_proj is W0, so the state with the pair off IS W0 with the pair off;
                   fp32 read noise only).  forbidden: about 0 (= 1 - G_abl / G, the gain survives the pair off).
  band             W0: nan (no G).  every state: measured in the same job from 20 random pairs, never filed as a number.
  forced_reading   W0 "no gain".  native copy: "off 1" (0.72), so the rule separates a native from a forced state.
                   forced: "net 1".  forbidden: "off 1".
  forbidden_reading  W0 "no gain".  native copy "above band" (0.72).  forced "above band" (1).  forbidden "within band".
  matched          W0: False (G = 0).  native against itself: True.  forced / forbidden: True, or "cap" (a reading).
  reproduced       stage 0 re-reads of E97's W0 falls and native G, own share: True (the same code and inputs).
  clause_ii        W0 only; expected True (the pair moves W0's lowercase-minus-capitalised gap on cased rows too).
  spared           native lookup net share at the copy pair <= 0.2: True (0.014, 0.003).

Edge cases, each run on CPU before filing (`~/commute-data/scratch/definition-lead/e99-writer/edge.py`):
  zero gain        net_share -> nan; forced/forbidden_reading -> "no gain"; matched -> False; a do-nothing state passes
                   no bar.
  negative gain    net_share -> nan (a share of a negative gain is undefined), reading "no gain".
  negative W0 fall net_share is still (fall - w0_fall) / G: a W0 that rises with the part off (lookup, -0.10) adds to the
                   net; tested at -0.5.
  ties             |net| == band reads "within band" (a tie is not above); |move| == band does not "move" (strict);
                   nearest() picks the earlier step on equal distance. |net - 1| <= tol and |G / target - 1| <= rel pass an
                   exact float tie, but the decimal ties 1.05 and 1.02 fail by rounding (1.05 - 1 > 0.05 in floats): run.
  empty windows    band([]) and band of all-nan draws -> nan; forbidden_reading with nan band -> "no band"; clause_ii with
                   a nan band -> False; a nan read anywhere -> the rule's nan branch, never a pass.
  one draw         band([x]) = |x|.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from commute.copying import finite
from commute.removal import band_of

NO_GAIN, NET_ONE, OFF_ONE = "no gain", "net 1", "off 1: the apparatus is wrong"
WITHIN, ABOVE, BELOW, NO_BAND = "within band: not through the part", "above band: through the part", "below band", "no band"


def net_share(fall: float, w0_fall: float, G: float) -> float:
    """(fall - w0_fall) / G; nan when G is not positive or any input is not finite."""
    if not all(finite(x) for x in (fall, w0_fall, G)) or G <= 0:
        return math.nan
    return (fall - w0_fall) / G


def band(draws: Sequence[float]) -> float:
    """The 95th percentile of |net share| over the finite random-pair draws; nan if none (E90's `band_of`)."""
    return band_of(draws)


def forced_reading(net: float, tol: float) -> str:
    if not finite(net):
        return NO_GAIN
    return NET_ONE if abs(net - 1) <= tol else OFF_ONE


def forbidden_reading(net: float, b: float) -> str:
    if not finite(net):
        return NO_GAIN
    if not finite(b):
        return NO_BAND
    return WITHIN if abs(net) <= b else ABOVE if net > 0 else BELOW


def matched(G: float, target: float, rel: float) -> bool:
    return finite(G) and finite(target) and target > 0 and abs(G / target - 1) <= rel


def nearest(prev: tuple[int, float, Any] | None, cur: tuple[int, float, Any], target: float) -> tuple[int, float, Any]:
    """Of the last check under the target and the first at or past (1 - rel) of it, the one whose G is nearer; ties to the
    earlier step."""
    if prev is None or not finite(prev[1]):
        return cur
    return prev if abs(prev[1] - target) <= abs(cur[1] - target) else cur


def match_status(G: float, target: float, rel: float, capped: bool) -> str:
    if matched(G, target, rel):
        return "matched"
    return "cap: not matched within the step cap" if capped else "nearest check outside the band of the target"


def reproduced(x: float, ref: float, tol: float) -> bool:
    return finite(x) and finite(ref) and finite(tol) and abs(x - ref) <= tol


def moves(move: float, b: float) -> bool:
    """A W0 move counts only strictly above its random-pair band."""
    return finite(move) and finite(b) and abs(move) > b


def clause_ii(cased_move: float, cased_band: float, lower_move: float, lower_band: float) -> bool:
    """R-net's admissibility clause (ii): the part moves W0's gap on cased rows (never trained on) and on lowercased rows."""
    return moves(cased_move, cased_band) and moves(lower_move, lower_band)


def spared(net: float, bar: float) -> bool:
    """A control is spared when its net share is at most `bar` (a rise counts as spared)."""
    return finite(net) and net <= bar


def restored_ok(G_restored: float, G_native: float, rel: float) -> bool:
    return matched(G_restored, G_native, rel)


def pace(step: int, native_step: int, at_native: float, target: float, status: str, rel: float, slow: float) -> str:
    """Forbidden arms' learning speed: as fast if matched by the native's step or within `rel` of the target there;
    slower by more than `slow` x if the match took over slow x the native's steps or was never reached."""
    if status.startswith("cap"):
        return f"slower by more than {slow:g}x: the cap was reached"
    if step <= native_step or (finite(at_native) and finite(target) and target > 0 and at_native >= (1 - rel) * target):
        return "as fast"
    return f"slower by more than {slow:g}x" if step > slow * native_step else "slower"
