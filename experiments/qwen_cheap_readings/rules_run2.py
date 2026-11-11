"""The rules of P3.6 run 2 (results/predictions_run2.json), pure, no torch. Descriptive: no reuse/new label on any update,
and the overlap is a description (on the 67M the check that a high overlap means "the same neurons firing harder" failed).

A floor is the n draws of one random change at an update's own per-matrix norms. The overlap's scale is the floor's distance
from the unchanged model's 1.0 ("room", 1 - mean); on_both's scale is the floor's mean.

spread          (max - min) of the draws over the scale (relative). Under two draws, a nan, or a zero scale: None.
band            stage 1: the largest spread over the reference updates' floors; None when no reference is readable.
reproducible    one update's floor: "reproducible" if its spread <= band, "not reproducible" if over; a zero scale reads "floor
                at ceiling" (overlap) or "floor at zero" (on_both); a nan or no band reads "unreadable".
vs_floor        one update against its floor, d = reading - mean, tol = band * scale: "below floor" if -d > tol, "above floor"
                if d > tol, else "at floor" (a gap equal to tol is at floor; the do-nothing update reads 1.0 against a floor
                of 1.0, so "at floor"). A nan reading, a nan draw or no band: "unreadable".
rel_gap         (reading - mean) / scale, the figure's unit; None where the scale is 0 or a value is nan.
call            the card's call, `rules.svd_call` over the 19 vs_floor reads: "rival A" if more than half of the readable read
                "below floor", "rival B" if more than half read "at floor", else "no call".
contrast        E92's `qwen_comparators.separates` on two seeds a side, and `ratio` = (mean a - mean b) / the larger seed
                range, clipped to +-CLIP (a zero range with a gap reads +-CLIP; no gap reads 0); a one-seed side is "unread".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from commute.qwen_comparators import separates

CLIP = 10.0
PAIRS = {"anchor 1": ("a1_s0", "a1_s1"), "anchor 2": ("a2_s42", "a2_s142"), "anchor 2 long": ("a2long_s0", "a2long_s1"),
         "RL": ("R_s0", "R_s70"), "RL null": ("N_s0", "N_s70"), "known pool": ("known_s0", "known_s1"),
         "never-solved pool": ("new_s0", "new_s1")}  # fmt: skip
CONTRASTS = {"RL vs shuffled-reward null": ("RL", "RL null"), "known vs never-solved pool": ("known pool", "never-solved pool")}
FLOORED = ("erank", "overlap", "on_both")  # the readings that have a matched random floor; real vs floor reads these


def _nan(*xs: float | None) -> bool:
    return any(x is None or math.isnan(x) for x in xs)


def scale(draws: Sequence[float], kind: str) -> float:
    mean = sum(draws) / len(draws)
    return 1.0 - mean if kind == "overlap" else mean


def spread(draws: Sequence[float], kind: str) -> float | None:
    if len(draws) < 2 or _nan(*draws) or scale(draws, kind) == 0.0:
        return None
    return (max(draws) - min(draws)) / scale(draws, kind)


def band(refs: Mapping[str, Sequence[float]], kind: str) -> float | None:
    got = [s for d in refs.values() if (s := spread(d, kind)) is not None]
    return max(got) if got else None


def reproducible(draws: Sequence[float], b: float | None, kind: str) -> str:
    if b is None or _nan(*draws):
        return "unreadable"
    if scale(draws, kind) == 0.0:
        return "floor at ceiling" if kind == "overlap" else "floor at zero"
    s = spread(draws, kind)
    return "unreadable" if s is None else "reproducible" if s <= b else "not reproducible"


def vs_floor(x: float, draws: Sequence[float], b: float | None, kind: str) -> str:
    if b is None or _nan(x, *draws):
        return "unreadable"
    d, tol = x - sum(draws) / len(draws), b * scale(draws, kind)
    return "below floor" if -d > tol else "above floor" if d > tol else "at floor"


def rel_gap(x: float, draws: Sequence[float], kind: str) -> float | None:
    if _nan(x, *draws) or scale(draws, kind) == 0.0:
        return None
    return (x - sum(draws) / len(draws)) / scale(draws, kind)


def ratio(a: Sequence[float | None], b: Sequence[float | None]) -> float | None:
    if len(a) != 2 or len(b) != 2 or _nan(*a, *b):
        return None
    x, y = [float(v) for v in a if v is not None], [float(v) for v in b if v is not None]
    gap, sd = (sum(x) - sum(y)) / 2, max(abs(x[0] - x[1]), abs(y[0] - y[1]))
    if sd == 0.0:
        return 0.0 if gap == 0.0 else math.copysign(CLIP, gap)
    return max(-CLIP, min(CLIP, gap / sd))


def contrast(a: Sequence[float | None], b: Sequence[float | None]) -> dict[str, object]:
    return {"a": list(a), "b": list(b), "verdict": separates(a, b), "ratio": ratio(a, b)}


def contrasts(per: Mapping[str, Mapping[str, float | None]], readings: Sequence[str]) -> dict[str, dict[str, object]]:
    """The two arm contrasts on every reading, and real against floor on each seeded pair's floored readings. `per[name]`
    holds each reading and, for a floored one, '<reading>_floor' (the mean of its draws)."""
    out: dict[str, dict[str, object]] = {}
    for title, (p, q) in CONTRASTS.items():
        out[title] = {r: contrast([per[n].get(r) for n in PAIRS[p]], [per[n].get(r) for n in PAIRS[q]]) for r in readings}
    for pair, names in PAIRS.items():
        out[f"real vs floor: {pair}"] = {r: contrast([per[n].get(r) for n in names], [per[n].get(f"{r}_floor") for n in names])
                                         for r in FLOORED}  # fmt: skip
    return out
