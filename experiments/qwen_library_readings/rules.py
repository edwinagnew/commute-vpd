"""The rules of E169 (results/predictions.json), pure, no torch. Descriptive: no reuse or new label on any update, no
share, and no joined score; each reading of each pair is read on its own.

A pair's gap at one seed is side A's reading minus side B's at that seed (the twins: real minus shuffled).

floor_band   per pair and floored reading (steering, effective rank, overlap): the readings of random changes at the
             per-matrix norms of the set's smallest- and largest-norm states (the references), on side A's rows (FA) and
             side B's rows (FB). band = the largest |mean FA[i] - mean FB[j]| over the reference pairs (i, j), plus the
             largest draw range (max - min) of any one list: the gap that the rows and any size in the set's range give
             a random change. None when a list has under two draws or holds a None or nan.
floored      a floored reading's cell from its seed gaps and band: "unreadable" (under two gaps, a None or nan, or no
             band); "beyond floor" if every gap is further from 0 than the band and all share one sign; "inside floor"
             if |mean gap| <= band (a gap equal to the band, to TIE relative, is inside; the do-nothing pair reads 0
             against any band: inside); else "mixed".
signed       a reading without a floor (size, both priors): "unreadable" as above; "A higher" if every gap > 0, "B higher"
             if every gap < 0, else "mixed" (a gap of exactly 0 is mixed).
card_call    at the late save, the two built pairs: "rival A" if each has at least one floored reading "beyond floor";
             "rival B" if every floored reading of both is "inside floor" and both size cells read one sign; else
             "no call". Recorded beside the scorecard as a description (the card is descriptive), never a label.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

FLOORED = ("steer", "erank", "overlap")
SIGNED = ("size", "prior_rank", "prior_logp")
BUILT = ("copy_cipher", "facts")
SEEDS = (0, 1, 2)
TIE = 1e-9  # relative: a mean gap equal to the band up to float rounding (3 x 0.1 / 3 > 0.1) reads as equal


@dataclass(frozen=True)
class Pair:
    a: str  # the library's arm names, side A minus side B
    b: str
    run: str  # the library's results folder (experiments/qwen_pair_library/results/<run>/<arm>_s<seed>.json)
    prefix: str  # the W&B artifact type of its states
    saves: tuple[str, ...]  # "last" (the late save) and "5x" (the matched save), as the library labels them


PAIRS = {"copy_cipher": Pair("copy", "cipher", "library_run3", "p35r3", ("last", "5x")),
         "facts": Pair("known", "invented", "library_run3", "p35r3", ("last", "5x")),
         "gold": Pair("gold", "gold_shuf", "library", "p35", ("last", "5x")),
         "template": Pair("template", "template_shuf", "library", "p35", ("last",))}  # no 5x save: fails the fit rule


def _bad(xs: Sequence[float | None]) -> bool:
    return any(x is None or math.isnan(x) for x in xs)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def floor_band(FA: Mapping[str, Sequence[float | None]], FB: Mapping[str, Sequence[float | None]]) -> float | None:
    lists = [*FA.values(), *FB.values()]
    if not FA or not FB or any(len(v) < 2 or _bad(v) for v in lists):
        return None
    clean = [[float(x) for x in v if x is not None] for v in lists]
    a, b = [_mean(v) for v in clean[: len(FA)]], [_mean(v) for v in clean[len(FA) :]]
    return max(abs(x - y) for x in a for y in b) + max(max(v) - min(v) for v in clean)


def floored(gaps: Sequence[float | None], band: float | None) -> str:
    if len(gaps) < 2 or _bad(gaps) or band is None or math.isnan(band):
        return "unreadable"
    g, edge = [float(x) for x in gaps if x is not None], band * (1 + TIE)
    if all(abs(x) > edge for x in g) and (all(x > 0 for x in g) or all(x < 0 for x in g)):
        return "beyond floor"
    return "inside floor" if abs(_mean(g)) <= edge else "mixed"


def signed(gaps: Sequence[float | None]) -> str:
    if len(gaps) < 2 or _bad(gaps):
        return "unreadable"
    g = [float(x) for x in gaps if x is not None]
    return "A higher" if all(x > 0 for x in g) else "B higher" if all(x < 0 for x in g) else "mixed"


def card_call(cells: Mapping[str, Mapping[str, str]]) -> str:
    """`cells[pair][reading]` at the late save, as `floored` and `signed` read them."""
    if any(p not in cells for p in BUILT):
        return "no call"
    if all(any(cells[p].get(r) == "beyond floor" for r in FLOORED) for p in BUILT):
        return "rival A"
    inside = all(cells[p].get(r) == "inside floor" for p in BUILT for r in FLOORED)
    if inside and all(cells[p].get("size") in ("A higher", "B higher") for p in BUILT):
        return "rival B"
    return "no call"
