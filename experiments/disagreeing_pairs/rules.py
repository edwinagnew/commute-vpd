"""E151's filed rule (results/predictions.json -> definitions). Pure, no model.

Per pair, three readings, sides named as built (reuse minus new):
  prior     log10 prior gap, one value a side: its sign if |gap| >= prior_bar, else 0 ("matched").
  steering  strength-1 pooled steering share per seed, called by E143's rule (holes_fill rules.call).
  share     the re-gating share per seed, called by the same rule.
condition (read before the share): "opposite" iff the prior call is nonzero and steering "separates" with the other sign;
  "flat" iff the prior call is 0 and steering reads "does not" or "inert"; else "condition not met" (steering unreadable too).
reading: opposite -> "follows prior" / "follows steering" (the share separates with that sign), "neither" (the share reads
  "does not" or "inert"), "unreadable"; flat -> "separates, reuse side higher" / "separates, new side higher" / "does not"
  ("inert" counts as "does not"), "unreadable"; condition not met -> "not counted".
outcome over the counted opposite pairs (unreadable left out; pair C never counted): "prior" / "steering" / "neither" iff
  every one reads that, "split" otherwise, "no call" iff none.

Amendment 1 (results/predictions_amend1.json): a pair whose built premise is not met (pair B when the common-word
cipher's realised prior is under b_prior_min) reads "condition not met" whatever its prior and steering; the common-word
table is drawn by a ladder of filters (`filters`, `rung_ok`, `assign`, `window_prior`, `take_rung`). Every input is a realised value: the priors
of the built windows and the steering read on the built states.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from commute.jobs import load_experiment

CALL = load_experiment("holes_fill", "rules").call
EPS = 1e-12  # the inclusive prior bar gets the same float slack as E143's smallest gap
COUNTED = ("A_copy_common", "B_elicit_common", "D_rank1_oneword")  # pair C is reported beside, never counted
FOLLOWS = {"follows prior": "prior", "follows steering": "steering", "neither": "neither"}


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def prior_call(gap: float, bar: float) -> int:
    return sign(gap) if abs(gap) >= bar - EPS else 0


def condition(prior: int, steer: Mapping[str, Any]) -> str:
    if prior != 0 and steer["call"] == "separates" and sign(steer["gap"]) == -prior:
        return "opposite"
    if prior == 0 and steer["call"] in ("does not", "inert"):
        return "flat"
    return "condition not met"


def reading(cond: str, prior: int, steer: Mapping[str, Any], share: Mapping[str, Any]) -> str:
    if cond == "condition not met":
        return "not counted"
    if share["call"] == "unreadable":
        return "unreadable"
    sep = share["call"] == "separates"
    if cond == "flat":
        return ("separates, reuse side higher" if share["gap"] > 0 else "separates, new side higher") if sep else "does not"
    if not sep:
        return "neither"
    return "follows prior" if sign(share["gap"]) == prior else "follows steering" if sign(share["gap"]) == sign(steer["gap"]) else "neither"


def outcome(readings: Sequence[str]) -> str:
    """The counted opposite pairs' readings (unreadable already left out)."""
    if not readings:
        return "no call"
    kinds = {FOLLOWS[r] for r in readings}
    return kinds.pop() if len(kinds) == 1 else "split"


def pair_calls(p: Mapping[str, Any], th: Mapping[str, float]) -> dict[str, Any]:
    """p = {"prior_gap": decades, "steering": (reuse seeds, new seeds), "premise": bool (Amendment 1; absent = True)}:
    the prior and steering calls and the condition."""
    gap, premise = p["prior_gap"], p.get("premise", True)
    pc, steer = prior_call(gap, th["prior_bar"]), CALL(*p["steering"], th)
    cond = condition(pc, steer) if premise else "condition not met"
    return {"prior": {"gap": gap, "call": pc}, "steering": steer, "premise": premise, "condition": cond}


# ----------------------------------------------------------------------------- Amendment 1: the common-word ladder


def cap_name(value_max: float) -> str:
    return f"P <= {value_max:g}"


def rank_name(r: int) -> str:
    return f"rank > {r}"


def ratio_name(m: float) -> str:
    return f"<= {m:g}x median"


def filters(G: Tensor, pool: list[int], ranks: Sequence[int], ratios: Sequence[float], value_max: float) -> dict[str, Tensor]:
    """Each filter alone, (keys, pool) bool. G: (keys, vocab) W0's geometric-mean P(v | M k). The cap: P <= value_max;
    rank > r: strictly under the key's r-th largest P over the vocabulary (past its top r, as run 1); m x median: P <= m x
    v's median over the keys."""
    P, med = G[:, pool], G.quantile(0.5, dim=0)[pool]
    out = {cap_name(value_max): P <= value_max}
    out |= {rank_name(r): P < G.topk(r, dim=1).values[:, -1:] for r in ranks}
    return out | {ratio_name(m): P <= m * med for m in ratios}


def rung_ok(f: Mapping[str, Tensor], min_rank: int, median_ratio: float, value_max: float) -> Tensor:
    """A rung's admissible (keys, pool): the cap, its rank filter and its median filter together."""
    return f[cap_name(value_max)] & f[rank_name(min_rank)] & f[ratio_name(median_ratio)]


def assign(P: Tensor, ok: Tensor) -> list[int | None]:
    """Per key (row), a column: keys in order of fewest admissible columns first (ties: row order), each taking its
    highest-P unused admissible column (ties: lower column). None where a key finds none left."""
    out: list[int | None] = [None] * P.shape[0]
    used: set[int] = set()
    for i in sorted(range(P.shape[0]), key=lambda r: int(ok[r].sum())):
        cols = sorted(ok[i].nonzero().flatten().tolist(), key=lambda j: -float(P[i, j]))
        j = next((c for c in cols if c not in used), None)
        if j is not None:
            used.add(j)
        out[i] = j
    return out


def est_prior(log_p: Sequence[float], n: Sequence[int]) -> float:
    """The filed prior from per-key pieces: exp of the window-weighted mean of log G_heldout(k, v_k) over the trained keys
    (= geo_rate over the trained keys' held-out windows, the filed estimator); nan with no windows."""
    total = sum(n)
    return math.exp(sum(a * b for a, b in zip(log_p, n, strict=True)) / total) if total else math.nan


def window_prior(LP: Tensor, wkeys: Sequence[int], value_of: Mapping[int, int | None]) -> float:
    """est_prior on the filed prior's own windows, one piece per window: LP (windows, vocab) W0's log P on the trained
    keys' held-out windows, wkeys each window's key, value_of the table. Window for window the realised prior
    (keydep.geo_rate); nan with no windows or an incomplete table (a window whose key has no value)."""
    v = [value_of[k] for k in wkeys]
    if any(x is None for x in v):
        return math.nan
    return est_prior(LP[torch.arange(len(v)), torch.tensor(v, dtype=torch.long)].tolist(), [1] * len(v))


def take_rung(rungs: Sequence[Mapping[str, Any]], prior_min: float) -> int | None:
    """The first rung whose table is complete and whose estimated prior is >= prior_min; None if none qualifies."""
    return next((i for i, r in enumerate(rungs) if r["complete"] and r["prior_est"] >= prior_min), None)


def b_premise(prior_common: float, b_prior_min: float) -> bool:
    """Pair B is counted only if the common-word cipher's realised prior is >= b_prior_min (Amendment 1)."""
    return prior_common >= b_prior_min


def score(pairs: Mapping[str, Mapping[str, Any]], th: Mapping[str, float]) -> dict[str, Any]:
    """pairs[p] = {"prior_gap", "steering": (reuse, new), "share": (reuse, new)} -> per pair calls, condition and reading;
    the outcome over the counted opposite pairs; the flat pairs beside."""
    out: dict[str, Any] = {}
    for p, x in pairs.items():
        rec = pair_calls(x, th)
        rec["share"] = CALL(*x["share"], th)
        rec["reading"] = reading(rec["condition"], rec["prior"]["call"], rec["steering"], rec["share"])
        out[p] = rec
    opposite = [p for p in COUNTED if p in out and out[p]["condition"] == "opposite" and out[p]["reading"] != "unreadable"]
    flat = {p: out[p]["reading"] for p in COUNTED if p in out and out[p]["condition"] == "flat"}
    return {"pairs": out, "counted_opposite": opposite, "outcome": outcome([out[p]["reading"] for p in opposite]), "flat": flat}


def n_meeting(conditions: Mapping[str, str]) -> int:
    """Stage 1's stop count: pairs A, B, D meeting any condition."""
    return sum(conditions.get(p) in ("opposite", "flat") for p in COUNTED)


def filed(s: Mapping[str, Any]) -> dict[str, bool]:
    """The filed predictions P1-P5 on a score."""
    cond = {p: r["condition"] for p, r in s["pairs"].items()}
    rd = {p: r["reading"] for p, r in s["pairs"].items()}
    A, B, D = COUNTED
    return {"P1": cond.get(A) == "opposite" and cond.get(B) == "flat" and cond.get(D) == "opposite",
            "P2": rd.get(A) == "follows steering", "P3": rd.get(D) == "follows prior",
            "P4": rd.get(B) == "separates, reuse side higher", "P5": s["outcome"] == "split"}  # fmt: skip
