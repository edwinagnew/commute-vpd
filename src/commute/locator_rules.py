"""E97's filed rules, pure, so every edge case runs on CPU (tests/test_locator_rules.py). The reads are `commute.locator`;
plan reports/plans/2026-11-05_plan_e97-locator.md. No function here knows which part anyone expects: the expected head
reaches `control_passes` as an argument, from the filed predictions.

The located rule, per part, on every seed: the task state's own fall with the part at W0's means is at least a times its
gain G, the same seed's control state falls at most `control_max` times its own gain, and the task's fall is at least
`ratio` times the control's, in nats. a comes from stage 1's control (`a_from`), never guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from commute.copying import finite
from commute.locate import ranked, top_head

LOCATED, NO_GAIN = "located", "unreadable: no gain"
NONE_FOUND = "none at head/MLP grain"


def frac(x: float, G: float) -> float:
    """x / G; nan when G is not positive or either is not finite (a zero update has no share)."""
    return x / G if finite(x) and finite(G) and G > 0 else math.nan


def row_se(x: Tensor, rows: Tensor) -> float:
    """Standard error of x's mean with reads clustered by source row (F4: windows overlap): sqrt(g / (g - 1) * sum over
    rows of (sum of x - mean)^2) / N. nan under two rows or with a non-finite read."""
    x = x.double()
    ids, inv = torch.unique(rows, return_inverse=True)
    if len(ids) < 2 or not bool(torch.isfinite(x).all()):
        return math.nan
    s = torch.zeros(len(ids), dtype=torch.float64).index_add_(0, inv, x - x.mean())
    g = len(ids)
    return float((g / (g - 1) * s.pow(2).sum()).sqrt() / len(x))


def mean_prob(lo: Tensor) -> float:
    """The mean probability over the reads (F5), not the probability at the mean log-odds."""
    return float(torch.sigmoid(lo.double()).mean())


def trunc2(x: float) -> float:
    return math.floor(x * 100) / 100 if finite(x) else math.nan


def top_part(scores: Mapping[str, float]) -> str | None:
    """The one highest finite score; a tie at the top, or nothing finite, names no part (E96's `top_head`)."""
    return top_head(scores)


def pair_name(a: str, b: str) -> str:
    return "+".join(sorted((a, b)))


def a_from(top_shares: Sequence[float]) -> float:
    """The bar a: the weakest copy seed's own-terms share at its top head, truncated to two decimals. nan if any seed has
    none or there is no seed; the caller stops on nan or a <= 0, since the rule is then degenerate."""
    return trunc2(min(top_shares)) if top_shares and all(finite(x) for x in top_shares) else math.nan


def nominees(rec: Mapping[str, Any], k: int) -> list[str]:
    """The parts the carry and write maps nominate on one state, unit "lo": the top k by carry, the top k heads by write,
    the block MLP of each MLP matrix in the top k by write, and the greedy pair. First-seen order, no repeats.
    E97 Amendment 1 adds the parts its edges nominate (`edge_rules.edge_nominees`), when the record holds them; E97's
    own records hold none, so its labels are unchanged."""
    carry = {n: v["lo"] for n, v in rec["carry"].items()}
    heads = {n: v["lo"] for n, v in rec["write"]["head"].items()}
    mats = {n: v["lo"] for n, v in rec["write"]["matrix"].items()}
    out = ranked(carry)[:k] + ranked(heads)[:k] + [f"mlp{m.split('.')[1]}" for m in ranked(mats)[:k] if ".mlp." in m]
    out += list(rec.get("edge_nominees", []))
    out += [rec["pair"]] if rec.get("pair") else []
    return list(dict.fromkeys(out))


def seed_reading(task: Mapping[str, float], ctl: Mapping[str, float], a: float, th: Mapping[str, Any]) -> str:
    """One seed of the located rule. `task` and `ctl` hold G and the own-terms fall in nats, unit "lo". A share past 1
    counts; a part whose removal raises the state is not located; ties pass."""
    if not all(finite(v) for v in (task["G"], ctl["G"])) or task["G"] <= 0 or ctl["G"] <= 0:
        return NO_GAIN
    if not (finite(a) and a > 0):
        return "unreadable: no bar a"
    if not (finite(task["fall"]) and finite(ctl["fall"])):
        return "unreadable: not finite"
    if task["fall"] < 0:
        return "not located: the part raises the state"
    if task["fall"] < a * task["G"]:
        return "not located: the task falls under a"
    if ctl["fall"] > th["control_max"] * ctl["G"]:
        return "not located: the control is not spared"
    if task["fall"] < th["ratio"] * ctl["fall"]:
        return "not located: under twice the control's fall"
    return LOCATED


def task_reading(readings: Sequence[str]) -> dict[str, Any]:
    """Every seed must read located; a seed that disagrees makes it not located; any unreadable seed makes it unreadable."""
    bad = sorted({r for r in readings if r.startswith("unreadable")})
    located = bool(readings) and all(r == LOCATED for r in readings)
    word = LOCATED if located else bad[0] if bad else "not located" if readings else "unreadable: no seed"
    return {"located": located, "reading": word, "per_seed": list(readings)}


def best_located(recs: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Of the located parts, the one whose weakest seed's share is largest; ties to the first name in sorted order."""
    found = {p: min(r["shares"]) for p, r in sorted(recs.items()) if r["located"] and r["shares"]}
    return max(found, key=lambda p: found[p]) if found else None


# ----------------------------------------------------------------------------- stage 0 and the controls


def upstream(n: str, block: int) -> bool:
    """A part whose output does not depend on block `block`'s MLP: heads at blocks <= block, MLPs at blocks < block."""
    b = int(n[3:]) if n.startswith("mlp") else int(n.split(",")[0])
    return b < block or (b == block and not n.startswith("mlp"))


def plant_misses(rec: Mapping[str, Any], key: str, block: int, th: Mapping[str, Any]) -> list[str]:
    """A one-matrix plant in `key`, block `block`'s MLP output: what reads other than the construction says. nan misses."""
    tol, out = th["reread_tol"], []
    near = lambda x, v: finite(x) and abs(x - v) <= tol
    for m, v in rec["write"]["matrix"].items():
        if not near(v["lo"], 1.0 if m == key else 0.0):
            out.append(f"write {m} {v['lo']}")
    for n, v in {**rec["write"]["head"], **rec["write"]["mlp"]}.items():
        if not near(v["lo"], 1.0 if n == f"mlp{block}" else 0.0):
            out.append(f"write {n} {v['lo']}")
    for n, v in rec["carry"].items():
        if (upstream(n, block) and not near(v["lo"], 0.0)) or (n == f"mlp{block}" and not near(v["lo"], 1.0)):
            out.append(f"carry {n} {v['lo']}")
    if not (finite(rec["write"]["G_all"]["lo"]) and abs(rec["write"]["G_all"]["lo"]) <= th["zero_tol"]):
        out.append(f"all reverted G {rec['write']['G_all']['lo']}")
    return out


def inheritance(carry: Mapping[str, Mapping[str, float]], block: int) -> float:
    """The largest carry share on a part downstream of block `block`'s MLP: what a pipe inherits from a write. nan if none."""
    down = [v["lo"] for n, v in carry.items() if not upstream(n, block) and n != f"mlp{block}" and finite(v["lo"])]
    return max(down) if down else math.nan


def can_nominate(floor: float, nominated: Sequence[float]) -> bool:
    """False when the inheritance floor exceeds the smallest nominated carry share (or nothing is readable)."""
    vals = [x for x in nominated if finite(x)]
    return finite(floor) and bool(vals) and floor <= min(vals)


def write_within(plant_share: float, lookup_shares: Sequence[float]) -> bool:
    """The hand write's own-terms share with the copy head off stays within the lookup's: at most its largest seed."""
    vals = [x for x in lookup_shares if finite(x)]
    return finite(plant_share) and bool(vals) and plant_share <= max(vals)


def control_passes(tops: Sequence[str | None], expected: str) -> bool:
    """Stage 1: the carry map's top part is the expected head on every copy seed. Empty or a tie fails."""
    return bool(tops) and all(t == expected for t in tops)


def zero_ok(G_all: Sequence[float], tol: float) -> bool:
    """All 24 matrices reverted reads G within `tol` of 0 on every state; nan or none read fails."""
    return bool(G_all) and all(finite(g) and abs(g) <= tol for g in G_all)
