"""E101's pure rules and set builders (experiments/carrying_jobs; plan reports/plans/2026-11-05_plan_e101-carrying-jobs.md).
No model here, so every edge case runs on CPU (tests/test_carrying_jobs.py). Stage 0 (`run.py`) and the scorer
(`score.py`) call these; the thresholds are in experiments/carrying_jobs/results/predictions.json. E102 stage B
(experiments/qwen_carrying/{stage_b,maximise_b}.py) reuses them on Qwen3-1.7B; the maximiser's test takes the model's
firing as a function, and the forced-choice judge's rules come last.

A neuron is a flat index l x 3072 + i over the 67M's four blocks, as `neuron_r`. Its *engaged value* on a task is the mean,
over the task's read windows, of W0's activation at the answer position as a quantile of that neuron's own activations on
Pile positions (`quantiles`). A set's *engaged share* is the median of its neurons' engaged values (`set_share`).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from commute import elicit as EL

D_MLP = 3072


def quantiles(sorted_sample: Tensor, values: Tensor) -> Tensor:
    """(n, P) per-neuron Pile activations sorted along dim 1, (n, W) values -> (n, W): the share of the neuron's Pile
    sample at or below each value (empirical CDF, right-continuous). 0 below the sample, 1 at or above its max."""
    assert sorted_sample.shape[0] == values.shape[0] and sorted_sample.shape[1] > 0
    idx = torch.searchsorted(sorted_sample.contiguous(), values.contiguous(), right=True)
    return idx.double() / sorted_sample.shape[1]


def set_share(engaged: Sequence[float], idx: Sequence[int]) -> float:
    """The median engaged value over a set (mean of the middle two when the set is even). An empty set has none."""
    if not idx:
        return math.nan
    return float(statistics.median(engaged[i] for i in idx))


def blocks_of(idx: Sequence[int], width: int = D_MLP) -> dict[int, int]:
    """Neurons per block, flat index l x width + i (width 6144 on Qwen3-1.7B, E102)."""
    out: dict[int, int] = {}
    for i in idx:
        out[i // width] = out.get(i // width, 0) + 1
    return out


def block_matched(idx: Sequence[int], exclude: set[int], gen: torch.Generator, width: int = D_MLP) -> list[int]:
    """A random set with the same number of neurons per block as `idx`, none in `exclude`, drawn with `gen`."""
    out: list[int] = []
    for b, n in sorted(blocks_of(idx, width).items()):
        pool = [b * width + i for i in range(width) if b * width + i not in exclude]
        assert len(pool) >= n, f"block {b}: {len(pool)} neurons left for {n}"
        out += [pool[j] for j in torch.randperm(len(pool), generator=gen)[:n].tolist()]
    return out


def band(carrying: float, draws: Sequence[float]) -> dict[str, Any]:
    """A state's carrying share against its random draws: excess over the draws' median, and above the band (over every
    draw). Needs at least two finite draws."""
    ok = [d for d in draws if math.isfinite(d)]
    if len(ok) < 2 or not math.isfinite(carrying):
        return {"excess": math.nan, "above_band": None, "draw_min": None, "draw_max": None}
    return {"excess": carrying - statistics.median(ok), "above_band": carrying > max(ok), "draw_min": min(ok),
            "draw_max": max(ok), "draw_median": statistics.median(ok)}


def row_rule(excess: Mapping[str, float], row: Mapping[str, Sequence[str]], min_gap: float) -> dict[str, Any]:
    """E51/E52's rule (`elicit.separation`) on a row's excesses, reuse side first. States missing from `excess` drop
    out; a non-finite excess makes the row unreadable, as the rule does."""
    a = [excess[n] for n in row["reuse"] if n in excess]
    b = [excess[n] for n in row["new"] if n in excess]
    return EL.separation(list(a), list(b), min_gap)


def consensus(sets: Sequence[Sequence[int]], n: int) -> list[int]:
    """The n neurons in most of `sets`, ties by their best rank in any set, then by index: a deterministic sample of a
    task's carrying neurons that favours the ones every seed found."""
    count: dict[int, int] = {}
    best: dict[int, int] = {}
    for s in sets:
        for r, i in enumerate(s):
            count[i] = count.get(i, 0) + 1
            best[i] = min(best.get(i, r), r)
    order = sorted(count, key=lambda i: (-count[i], best[i], i))
    assert len(order) >= n, f"{len(order)} neurons for a sample of {n}"
    return order[:n]


def neuron_id(salt: str, flat: int) -> str:
    return hashlib.sha256(f"{salt}:{flat}".encode()).hexdigest()[:12]


def mark(pieces: Sequence[str], at: int) -> str:
    """Decoded token pieces joined, the one at `at` wrapped in ⟦ ⟧."""
    return "".join(pieces[:at]) + "⟦" + pieces[at] + "⟧" + "".join(pieces[at + 1 :])


# ----------------------------------------------------------------------------- stage 1


def outcome(examples: Sequence[float], near: Sequence[float], top: float, frac: float) -> str | None:
    """The essay's test on one neuron: an example or near miss fires strongly at >= frac of the neuron's strongest Pile
    activation `top`. 'win' if more examples than near misses fire strongly, 'tie' if as many, 'lose' if fewer. None
    (unscorable) when top <= 0 or either list is empty."""
    if top <= 0 or not examples or not near:
        return None
    se, sn = sum(x >= frac * top for x in examples), sum(x >= frac * top for x in near)
    return "win" if se > sn else "tie" if se == sn else "lose"


def rate(hits: Sequence[bool]) -> tuple[float, int]:
    return (sum(hits) / len(hits), len(hits)) if hits else (math.nan, 0)


def var(p: float, n: int) -> float:
    return p * (1 - p) / n


def decide(gap: float, se: float, min_gap: float) -> str:
    """'above' iff gap >= min_gap and gap >= 2 se; 'below_or_level' iff gap < min_gap and gap <= 2 se (so equal rates
    of 0 or 1, se 0, read level, not no call); else 'no_call'."""
    if gap >= min_gap and gap >= 2 * se:
        return "above"
    return "below_or_level" if gap < min_gap and gap <= 2 * se else "no_call"


def ok_side(p: float, n: int) -> bool:
    return n > 0 and math.isfinite(p)


def gap_rule(p1: float, n1: int, p0: float, n0: int, min_gap: float) -> dict[str, Any]:
    """Is the rate p1 (of n1) above the floor p0 (of n0)? `decide` on gap = p1 - p0, se its binomial sd. 'unreadable'
    when a side is empty or not finite."""
    rec = {"p1": p1, "n1": n1, "p0": p0, "n0": n0}
    if not (ok_side(p1, n1) and ok_side(p0, n0)):
        return {"verdict": "unreadable", **rec}
    gap, se = p1 - p0, math.sqrt(var(p1, n1) + var(p0, n0))
    return {"verdict": decide(gap, se, min_gap), "gap": gap, "se": se, **rec}


def row_kind(reuse: Mapping[str, Any], new: Mapping[str, Any], min_gap: float) -> dict[str, Any]:
    """P2's row. Each side is {'p', 'n', 'floor', 'n_floor'}: its related share and its own description's random floor.
    x = p - floor per side; `decide` on x_reuse - x_new, se from all four binomial variances (the two floors share the
    random pool, so this se is conservative when the floors are positively correlated)."""
    sides = {"reuse": dict(reuse), "new": dict(new)}
    if not all(ok_side(s["p"], s["n"]) and ok_side(s["floor"], s["n_floor"]) for s in sides.values()):
        return {"verdict": "unreadable", **sides}
    xr, xn = reuse["p"] - reuse["floor"], new["p"] - new["floor"]
    se = math.sqrt(sum(var(s["p"], s["n"]) + var(s["floor"], s["n_floor"]) for s in sides.values()))
    return {"verdict": decide(xr - xn, se, min_gap), "gap": xr - xn, "se": se, "x_reuse": xr, "x_new": xn, **sides}


def load_texts(folder: Path, batches: Mapping[str, Sequence[str]], n_ex: int, n_near: int) -> tuple[dict[str, dict[str, list[str]]], list[str], list[str]]:
    """The maximiser agents' files, one per batch (`folder/<batch>.json`, {"batch", "texts": {id: {"examples",
    "near_misses"}}}): (id -> texts, batches whose file is missing or unreadable, ids dropped). An id is dropped when it is
    not in that batch, or its lists are not n_ex / n_near non-blank strings. A dropped or missing id has no scores, so
    the scorer reads it as unscorable."""
    texts: dict[str, dict[str, list[str]]] = {}
    missing: list[str] = []
    bad: list[str] = []

    def good(xs: Any, n: int) -> bool:
        return isinstance(xs, list) and len(xs) == n and all(isinstance(x, str) and x.strip() for x in xs)

    for b, ids in batches.items():
        try:
            got = json.loads((folder / f"{b}.json").read_text())["texts"]
            assert isinstance(got, dict)
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            missing.append(b)
            continue
        for i, e in got.items():
            e = e if isinstance(e, dict) else {}
            if i in ids and good(e.get("examples"), n_ex) and good(e.get("near_misses"), n_near):
                texts[i] = {"examples": e["examples"], "near_misses": e["near_misses"]}
            else:
                bad.append(i)
    return texts, missing, bad


def contrast_top(engaged: Mapping[str, Tensor], task: str, k: int, exclude: set[int]) -> tuple[list[int], list[str]]:
    """Amendment 1's task-specific controls: the k neurons, none in `exclude`, whose engaged value on `task` most exceeds
    their mean engaged value over the other tasks (ties by index). A task whose engaged vector equals another's (the same
    windows with another target: W0 fires the same at the answer position) counts once, and `task`'s twins not at all.
    Returns the neurons and the tasks the mean is over."""
    own, others = engaged[task], {}
    for n, v in engaged.items():
        if not torch.equal(v, own) and not any(torch.equal(v, o) for o in others.values()):
            others[n] = v
    assert others, f"{task}: no task with other windows"
    c = own - torch.stack(list(others.values())).mean(dim=0)
    top = [i for i in torch.sort(c, descending=True, stable=True).indices.tolist() if i not in exclude][:k]
    assert len(top) == k, f"{task}: {len(top)} neurons for {k}"
    return top, list(others)


def every_task(verdicts: Sequence[str]) -> str:
    """Amendment 1's C1a and C1b over the control tasks: 'pass' iff every task reads 'above'; 'fail' iff any reads
    'below_or_level'; else (no task, or a no_call or unreadable task and none below) 'no call'."""
    if verdicts and all(v == "above" for v in verdicts):
        return "pass"
    return "fail" if "below_or_level" in verdicts else "no call"


def self_test(outcomes: Sequence[str | None], min_win: float) -> dict[str, Any]:
    """C2: the share of 'win' among scorable neurons (outcome not None); ok iff it reaches min_win. No scorable neuron
    gives nan and not ok."""
    scorable = [o for o in outcomes if o is not None]
    share = sum(o == "win" for o in scorable) / len(scorable) if scorable else math.nan
    counts = {k: sum(o == k for o in outcomes) for k in ("win", "tie", "lose")} | {"unscorable": len(outcomes) - len(scorable)}
    return {"win_share": share, "n_scorable": len(scorable), "n": len(outcomes), "counts": counts, "ok": share >= min_win}


# ----------------------------------------------------------------------------- the maximiser's test, any model


Fire = Callable[[Tensor], Tensor]  # (B, n) token ids on the model's device -> (B, n, neurons) W0 firing, flat index


@torch.no_grad()
def peaks(fire: Fire, seqs: Sequence[Sequence[int]], flats: Sequence[int], batch: int, device: Any) -> list[float]:
    """Per sequence, the max over its tokens of W0's firing of neuron flats[k]. Right-padded batches (pad id 0); causal,
    so padding cannot change a real token's firing, and padded positions are masked out of the max."""
    out: list[float] = []
    for s in range(0, len(seqs), batch):
        chunk, fl = seqs[s : s + batch], torch.tensor(flats[s : s + batch], device=device)
        n, lens = max(len(q) for q in chunk), torch.tensor([len(q) for q in chunk], device=device)
        toks = torch.tensor([list(q) + [0] * (n - len(q)) for q in chunk], device=device)
        b, t = torch.arange(len(chunk), device=device), torch.arange(n, device=device)
        v = fire(toks)[b[:, None], t[None, :], fl[:, None]].float()
        v[t[None, :] >= lens[:, None]] = -torch.inf
        out += v.max(dim=1).values.cpu().tolist()
    return out


def own_windows(flats: Mapping[str, int], top: Mapping[str, Tensor], rows: Tensor, decode: Callable[[list[int]], str],
                encode: Callable[[str], list[int]], ctx: int, max_tokens: int) -> dict[str, list[list[int]]]:
    """id -> its corpus windows ranked 2-6 as token ids: the ±ctx slice around the peak, decoded as text and encoded again."""
    out = {}
    for i, f in flats.items():
        wins = []
        for r in range(1, 1 + 5):
            row, pos = rows[int(top["rows"][r, f])], int(top["pos"][r, f])
            wins.append(encode(decode(row[max(0, pos - ctx) : pos + ctx + 1].tolist()))[:max_tokens])
        out[i] = wins
    return out


def self_test_run(fire: Fire, neurons: Mapping[str, Mapping[str, Any]], wins: Mapping[str, list[list[int]]], seed: int,
                  frac: float, min_win: float, batch: int, device: Any, log: Callable[[str], None], others: int = 5) -> dict[str, Any]:
    """C2 (E101): per neuron, its own windows ranked 2-6 as examples against `others` other neurons' windows of the same
    ranks as near misses, scored by `outcome`; `self_test` over the outcomes. `neurons`: id -> {"flat", "top_act"}."""
    rng, ids, seqs, flats = random.Random(seed), sorted(neurons), [], []
    for i in ids:
        pick = rng.sample([j for j in ids if j != i], others)
        for r, w in enumerate(wins[i]):  # example r: own rank r+2; near miss r: another neuron's window of the same rank
            seqs += [w, wins[pick[r]][r]]
            flats += [neurons[i]["flat"]] * 2
    p, outs = peaks(fire, seqs, flats, batch, device), []
    for n, i in enumerate(ids):
        ex, nm = p[10 * n : 10 * n + 10 : 2], p[10 * n + 1 : 10 * n + 10 : 2]
        outs.append(outcome(ex, nm, neurons[i]["top_act"], frac))
        log(f"[C2 {i}] top {neurons[i]['top_act']:.4f} own {json.dumps([round(x, 4) for x in ex])} "
            f"other {json.dumps([round(x, 4) for x in nm])} -> {outs[-1]}")
    st = self_test(outs, min_win)
    log(f"[C2] {json.dumps(st)} (expect >= {min_win})")
    return st


def text_scores(fire: Fire, neurons: Mapping[str, Mapping[str, Any]], texts: Mapping[str, Mapping[str, list[str]]],
                encode: Callable[[str], list[int]], frac: float, max_tokens: int, batch: int, device: Any,
                log: Callable[[str], None]) -> dict[str, dict[str, Any]]:
    """The agents' texts: id -> {"examples": [peak], "near_misses": [peak], "outcome"}, each text's peak W0 firing of its
    neuron over its tokens, every score logged."""
    ids, seqs, flats = sorted(texts), [], []
    for i in ids:
        for t in texts[i]["examples"] + texts[i]["near_misses"]:
            seqs.append(encode(t)[:max_tokens])
            flats.append(neurons[i]["flat"])
    p, out, at = peaks(fire, seqs, flats, batch, device), {}, 0
    for i in ids:
        ne, nn = len(texts[i]["examples"]), len(texts[i]["near_misses"])
        ex, nm = p[at : at + ne], p[at + ne : at + ne + nn]
        at += ne + nn
        o = outcome(ex, nm, neurons[i]["top_act"], frac)
        out[i] = {"examples": ex, "near_misses": nm, "outcome": o}
        log(f"[score {i}] top {neurons[i]['top_act']:.4f} examples {json.dumps([round(x, 4) for x in ex])} "
            f"near {json.dumps([round(x, 4) for x in nm])} -> {o}")
    return out


# ----------------------------------------------------------------------------- E102 stage B: the forced-choice judge


def chance_rule(p: float, n: int, chance: float, min_gap: float) -> dict[str, Any]:
    """Is the rate p (of n) above a fixed chance rate? `decide` on p - chance, se p's binomial sd alone."""
    if not ok_side(p, n):
        return {"verdict": "unreadable", "p": p, "n": n, "chance": chance}
    gap, se = p - chance, math.sqrt(var(p, n))
    return {"verdict": decide(gap, se, min_gap), "gap": gap, "se": se, "p": p, "n": n, "chance": chance}


def forced_gate(p: float, n: int, p_rand: float, n_rand: int, chance: float, min_gap: float) -> dict[str, Any]:
    """CB1: a control group's rate for its own description against both the random neurons' rate for it (`gap_rule`)
    and chance (`chance_rule`): 'pass' iff both read 'above', 'fail' iff either reads 'below_or_level' (`every_task`),
    'unreadable' if either side is, else 'no call'."""
    rand, ch = gap_rule(p, n, p_rand, n_rand, min_gap), chance_rule(p, n, chance, min_gap)
    if "unreadable" in (rand["verdict"], ch["verdict"]):
        return {"status": "unreadable", "random": rand, "chance": ch}
    return {"status": every_task([rand["verdict"], ch["verdict"]]), "random": rand, "chance": ch}


def load_picks(folder: Path, batches: Mapping[str, Sequence[str]], letters: Mapping[str, Mapping[str, str]],
               fits: Sequence[str]) -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    """The forced-choice judge's files (`folder/<batch>.json`, {"batch", "picks": {id: {"letter", "fit"}}}): (id ->
    {"desc": the description's name under that batch's letters, "fit"}, batches missing or unreadable, ids dropped). An
    id is dropped when it is not in that batch, its letter is not one of the batch's, or its fit is not in `fits`."""
    picks: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    bad: list[str] = []
    for b, ids in batches.items():
        try:
            got = json.loads((folder / f"{b}.json").read_text())["picks"]
            assert isinstance(got, dict)
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            missing.append(b)
            continue
        for i, e in got.items():
            e = e if isinstance(e, dict) else {}
            if i in ids and e.get("letter") in letters[b] and e.get("fit") in fits:
                picks[i] = {"desc": letters[b][e["letter"]], "fit": e["fit"]}
            else:
                bad.append(i)
    return picks, missing, bad
