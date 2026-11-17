"""E155 (neuron_overlap, cell P2.18) filed rule, results/predictions.json → definitions. Pure, no model. Descriptive: no
verdict on reuse.

Per pair and reading, on each side's saved seeds (2 to 8 per side):
  gap = mean(side built to reuse) - mean(side built to be new);
  band = the larger of the two sides' max - min over seeds: the seed spread, measured on this job's own readings (stage 1),
         never a number set in advance;
  "unreadable"         fewer than 2 values on a side, or a nan among them;
  "reuse side higher"  gap > band (strict); on the overlap reading this is "the reuse side overlaps more";
  "new side higher"    -gap > band (strict);
  "no call"            otherwise. A gap equal to the band is no call; the do-nothing state on both sides (every value 1.0)
                       gives gap 0 and band 0, no call.
Floor, per side: the side's mean minus the reading of its random update (matched Frobenius norm per matrix to the side's first
seed); reported, not part of the call.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

CALLS = ("reuse side higher", "new side higher", "no call", "unreadable")


def call(reuse: Sequence[float], new: Sequence[float]) -> dict[str, Any]:
    vals = [*reuse, *new]
    if len(reuse) < 2 or len(new) < 2 or any(math.isnan(v) for v in vals):
        return {"call": "unreadable", "gap": math.nan, "band": math.nan}
    gap = statistics.fmean(reuse) - statistics.fmean(new)
    band = max(max(reuse) - min(reuse), max(new) - min(new))
    verdict = "reuse side higher" if gap > band else "new side higher" if -gap > band else "no call"
    return {"call": verdict, "gap": gap, "band": band}


def floor(side: Sequence[float], random: float) -> dict[str, Any]:
    """The side's mean against its random update's reading (nan in, nan out)."""
    m = statistics.fmean(side) if side else math.nan
    return {"mean": m, "random": random, "above": m - random}


def positive_control(scaled: float, random: float) -> bool:
    """The hand-built 'same neurons, harder' update reads above a random update of the same per-matrix size (strict)."""
    return scaled > random
