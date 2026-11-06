"""E109 (larger_size): the filed rules, with no model, so the CPU dry run and run.py's score stage read the same code.
Plan: reports/plans/2026-11-06_plan_e109-neurons-larger-size.md; rules results/predictions.json `power_gate`, `learned`,
`wrong_rule`.

A set is a state's top-1024 carrying neurons on one probe (`qwen_neurons.top_set`: the signed score's positive part, as
E98; on the answer-token probes also by |score|, key "carry_abs"). Overlaps are `qwen_neurons.overlap`: the share of the
first set inside the second, NaN if either is empty (the do-nothing state's set is empty on every probe). State names
end in their seed ("KNOWN_end42"), so E107's `pairs` reads them: `pairs(sets, "WRONG_end", "KNOWN_end")` is the four
overlaps WRONG_end42-KNOWN_end42, 42-142, 142-42, 142-142."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from commute import qwen_neurons as QN
from commute.jobs import load_experiment

S107: Any = load_experiment("label_vs_style", "score")  # E107's seeds, K, pairs and prefers_wrong
SEEDS, K, pairs = S107.SEEDS, S107.K, S107.pairs


def _sizes_ok(sets: Mapping[str, Sequence[int]], arms: Sequence[str]) -> bool:
    return all(len(sets[f"{a}{s}"]) == K for a in arms for s in SEEDS)


def power(w: float, x: float, ok_sizes: bool) -> dict[str, Any]:
    """The power gate on one probe. w: the known arm's seed overlap; x: the highest known-in-fictional overlap (four
    pairs); band = 1 - w. "power" iff w - x > band (strict), else "no power"; "unread" with w or x not finite or a set
    not of K neurons."""
    if not (QN.finite(w) and QN.finite(x) and ok_sizes):
        return {"verdict": "unread", "w": w, "x": x}
    band = 1 - w
    return {"verdict": "power" if w - x > band else "no power", "w": w, "x": x, "band": band, "margin": w - x - band}


def wrong_verdict(w_known: float, w_wrong: float, x: float, cross: Sequence[float], ok_sizes: bool) -> dict[str, Any]:
    """The filed rule on one probe. w = min(w_known, w_wrong), band = 1 - w: the less stable arm's seed noise, as E107's
    `wrong_verdict` (so a noisier wrong arm cannot read "familiarity" from noise alone).
    "familiarity" iff max(cross) <= x + band; "topic or style" iff min(cross) >= w - band and min(cross) - x > band
    (strict); "no call" otherwise; "unread" with a value not finite, not four cross values or a set not of K neurons.
    Both cannot hold: familiarity needs max(cross) <= x + band, topic or style min(cross) > x + band."""
    if len(cross) != 4 or not all(QN.finite(v) for v in (w_known, w_wrong, x, *cross)) or not ok_sizes:
        return {"verdict": "unread", "w_known": w_known, "w_wrong": w_wrong, "x": x, "cross": list(cross)}
    w, lo, hi = min(w_known, w_wrong), min(cross), max(cross)
    band = 1 - w
    v = "familiarity" if hi <= x + band else "topic or style" if lo >= w - band and lo - x > band else "no call"
    return {"verdict": v, "w_known": w_known, "w_wrong": w_wrong, "w": w, "x": x, "band": band, "cross": list(cross),
            "cross_min": lo, "cross_max": hi, "familiarity_needs_max": x + band, "style_needs_min": w - band,
            "style_needs_over": x + band}


def references(sets: Mapping[str, Sequence[int]], known: str = "KNOWN_end", fict: str = "FICT_learned") -> dict[str, Any]:
    """w, x and the power gate from the existing states' sets on one probe (stage 1, before a wrong arm trains)."""
    w = QN.overlap(sets[f"{known}142"], sets[f"{known}42"])
    xs = pairs(sets, known, fict)
    x = max(xs) if all(QN.finite(v) for v in xs) else float("nan")
    return {"w": w, "x_pairs": xs, "x": x, "gate": power(w, x, _sizes_ok(sets, (known, fict)))}


def learned(prefers: float, recall: float, min_prefers: float, min_recall: float) -> dict[str, Any]:
    """The "learned" gate for one wrong end state: holds iff the share of rows whose wrong sentence's content log P is
    above the true one's is at least min_prefers and the greedy answers' mean recall of the wrong answer's words is at
    least min_recall (both inclusive). A value not finite does not hold."""
    holds = QN.finite(prefers) and QN.finite(recall) and prefers >= min_prefers and recall >= min_recall
    return {"holds": bool(holds), "prefers_wrong": prefers, "recall": recall, "min_prefers": min_prefers, "min_recall": min_recall}


def wrong_rule(sets: Mapping[str, Sequence[int]], gates: Sequence[bool], known: str = "KNOWN_end",
               fict: str = "FICT_learned", wrong: str = "WRONG_end") -> dict[str, Any]:
    """The rule on one probe from the sets keyed by state name. "scored" only when both wrong seeds passed the learned
    gate (`gates`, one bool a seed) and the probe has power; otherwise the verdict is computed and labelled "premise
    failed" or "no power at this size". Reported beside: the wrong seeds' overlap, the four wrong-in-fictional and the
    four known-in-fictional overlaps."""
    ref = references(sets, known, fict)
    w_wrong = QN.overlap(sets[f"{wrong}142"], sets[f"{wrong}42"])
    out = wrong_verdict(ref["w"], w_wrong, ref["x"], pairs(sets, wrong, known), _sizes_ok(sets, (known, wrong)))
    status = ("premise failed" if not (len(gates) == 2 and all(gates)) else
              "no power at this size" if ref["gate"]["verdict"] != "power" else "scored")
    return out | {"status": status, "power": ref["gate"], "known_vs_fict": ref["x_pairs"], "wrong_vs_fict": pairs(sets, wrong, fict)}


def sentences(true: Sequence[float], wrong: Sequence[float]) -> dict[str, Any]:
    """Per-row content log P on the true and the wrong sentence: E107's share of rows whose wrong sentence is strictly
    above (a tie is not) and the mean true-minus-wrong gap (nats a token). No rows or unequal lengths: NaN."""
    if not true or len(true) != len(wrong):
        return {"prefers_wrong": float("nan"), "gap": float("nan"), "rows": len(true)}
    return {"prefers_wrong": S107.prefers_wrong(true, wrong), "gap": statistics.mean(t - w for t, w in zip(true, wrong, strict=True)),
            "rows": len(true)}
