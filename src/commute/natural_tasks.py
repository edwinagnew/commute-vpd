"""E124's natural task, pronoun only since the pre-run revision (plan reports/plans/2026-11-06_plan_e124-natural-
definition-tasks.md, "Revision before any run"; appendix §2): windows cut from Pile rows where the text's next word is
" he" or " she" and exactly one pool name sits 3-15 tokens back, the name list and pool, and the probe positions.
Agreement was dropped (155 pattern windows per 32k rows) and its builders removed. No model is read here;
tests/test_natural_tasks.py runs the pieces on hand-made rows.

A record is one (row, t), t the prediction position (the last token the model sees): `content` the name token, `c` = C
(1: male, answer word A " he"), `kind` = c (the gender cell windows are balanced over) and `next` the text's token at
t + 1 (the natural member's answer).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

Rec = dict[str, Tensor]  # row, t, content, c, kind, next: (n,) int64 each


def single_ids(encode: Callable[[str], list[int]], words: Sequence[str]) -> list[int]:
    ids = [encode(w) for w in words]
    assert all(len(i) == 1 for i in ids), f"not single tokens: {[w for w, i in zip(words, ids, strict=True) if len(i) != 1]}"
    return [i[0] for i in ids]


def pronoun_matches(rows: Tensor, names: Tensor, near: int, far: int, half: int, nexts: Tensor | None,
                    chunk: int = 32768) -> dict[str, Tensor]:  # fmt: skip
    """(row, t, name) where the only pool name in [t - far, t] sits in [t - far, t - near], t in [max(half, far), T - 1),
    and the token at t + 1 is in `nexts` (None: any). Rows are read `chunk` at a time (memory)."""
    out = [_matches(rows[s : s + chunk], names, near, far, half, nexts) for s in range(0, rows.shape[0], chunk)]
    for i, s in enumerate(range(0, rows.shape[0], chunk)):
        out[i]["row"] += s
    return {k: torch.cat([o[k] for o in out]) for k in ("row", "t", "name")}


def _matches(rows: Tensor, names: Tensor, near: int, far: int, half: int, nexts: Tensor | None) -> dict[str, Tensor]:
    nm = torch.isin(rows, names).int()
    cs = torch.cat([torch.zeros(rows.shape[0], 1, dtype=torch.int), nm.cumsum(1, dtype=torch.int)], 1)  # names in [0, j)
    lo, hi = max(half, far), rows.shape[1] - 1
    t = torch.arange(lo, hi)
    ok = (cs[:, t + 1] - cs[:, t - far] == 1) & (cs[:, t + 1] - cs[:, t - near + 1] == 0)
    if nexts is not None:
        ok &= torch.isin(rows[:, lo + 1 : hi + 1], nexts)
    r, j = torch.nonzero(ok, as_tuple=True)
    tt = j + lo
    span = tt[:, None] - far + torch.arange(far - near + 1)
    p = tt - far + nm[r[:, None], span].argmax(1)
    return {"row": r, "t": tt, "name": rows[r, p]}


def pronoun_records(m: Mapping[str, Tensor], rows: Tensor, male: Mapping[int, int]) -> Rec:
    c = torch.tensor([male[int(n)] for n in m["name"]], dtype=torch.long)
    return {"row": m["row"], "t": m["t"], "content": m["name"], "c": c, "kind": c.clone(), "next": rows[m["row"], m["t"] + 1]}


def subset(rec: Rec, keep: Tensor) -> Rec:
    return {k: v[keep] for k, v in rec.items()}


def agree_mask(rec: Rec, a: int, b: int) -> Tensor:
    """The drop rule: True where the text's next token is the answer C gives (a when c = 1, b when c = 0)."""
    return torch.where(rec["c"] == 1, rec["next"] == a, rec["next"] == b)


def drop_shares(rec: Rec, a: int, b: int) -> dict[str, float]:
    """The share of matches the drop rule removes, per kind and overall (nan for an empty kind)."""
    bad = ~agree_mask(rec, a, b)
    out = {f"kind{k}": float(bad[rec["kind"] == k].double().mean()) if bool((rec["kind"] == k).any()) else float("nan") for k in (0, 1)}
    return {**out, "all": float(bad.double().mean()) if len(bad) else float("nan")}


def sample_cells(rec: Rec, cells: Sequence[tuple[int, int]], n: int, seed: int) -> tuple[Rec, dict[str, int]]:
    """n records, n / len(cells) from each (c, kind) cell, drawn seeded without replacement, in record order. Returns the
    sample and each cell's available count; a short cell gives what it has (the caller's count gate reads it)."""
    gen = torch.Generator().manual_seed(seed)
    each, picks, avail = n // len(cells), [], {}
    assert each * len(cells) == n, f"{n} windows do not split over {len(cells)} cells"
    for c, k in cells:
        idx = torch.nonzero((rec["c"] == c) & (rec["kind"] == k), as_tuple=True)[0]
        avail[f"c{c}_kind{k}"] = len(idx)
        picks.append(idx[torch.randperm(len(idx), generator=gen)[:each]])
    keep = torch.cat(picks).sort().values
    return subset(rec, keep), avail


def windows(rows: Tensor, rec: Rec, half: int, extra: int = 0) -> Tensor:
    """(n, half + 1 + extra) tokens rows[row, t - half : t + 1 + extra]: the prediction position is the column `half`;
    `extra` = 1 adds the text's next token as a last column (for the next-token loss)."""
    return rows[rec["row"][:, None], rec["t"][:, None] - half + torch.arange(half + 1 + extra)]


def cap_and_balance(rec: Rec, group: Tensor, cap: int, cells: Sequence[tuple[int, int]], seed: int) -> Rec:
    """Probe positions: at most `cap` per group id (`case_label.cap_groups`), then each (c, kind) cell cut, seeded, to
    the smallest cell's size, so the classes are balanced."""
    from commute.case_label import cap_groups

    rec = subset(rec, cap_groups(group, cap, seed))
    n = min(int(((rec["c"] == c) & (rec["kind"] == k)).sum()) for c, k in cells)
    out, _ = sample_cells(rec, cells, n * len(cells), seed + 1)
    return out


def ssa_names(path: Path, years: tuple[int, int], share: float) -> dict[str, int]:
    """Names from Hadley Wickham's SSA extract (columns year, name, percent, sex): percent summed over the years
    (inclusive) per sex; a name whose larger side holds >= `share` of its sum gets 1 (boy) or 0 (girl), else is out."""
    tot: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if years[0] <= int(row["year"]) <= years[1]:
                s = tot.setdefault(row["name"], [0.0, 0.0])
                s[row["sex"] == "boy"] += float(row["percent"])
    return {n: int(b >= g) for n, (g, b) in tot.items() if max(g, b) >= share * (g + b) and g + b > 0}


def name_pool(names: Mapping[str, int], encode: Callable[[str], list[int]], counts: Tensor, per_gender: int,
              exclude: set[int]) -> tuple[dict[int, int], list[str]]:  # fmt: skip
    """{token: male} for the `per_gender` names of each gender whose " Name" is one token, most frequent in `counts`
    (a (V,) count per token id) first, ties to the lower id; absent and `exclude`d tokens out. The revision's counted
    filter comes first: a name whose lowercase " name" is one token with a count >= its " Name" count is out (months,
    places, common words); those names are returned, sorted. Two names on one token keep the first in `names`' order."""
    toks: dict[int, int] = {}
    out: list[str] = []
    for name, male in names.items():
        ids, low = encode(" " + name), encode(" " + name.lower())
        if len(ids) != 1 or ids[0] in exclude or counts[ids[0]] == 0:
            continue
        if len(low) == 1 and low[0] != ids[0] and counts[low[0]] >= counts[ids[0]]:
            out.append(name)
            continue
        toks.setdefault(ids[0], male)
    pool: dict[int, int] = {}
    for g in (1, 0):
        ranked = sorted((t for t, m in toks.items() if m == g), key=lambda t: (-int(counts[t]), t))[:per_gender]
        pool.update({t: g for t in ranked})
    return pool, sorted(out)


def split_within(groups: Sequence[Sequence[int]], frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Per group (one gender), `case_label.split_pairs` on one-token "pairs" with round(frac x size) to train."""
    from commute.case_label import split_pairs

    tr, he = [], []
    for g in groups:
        a, b = split_pairs([(t, t) for t in g], round(frac * len(g)), seed)  # a name is a pair of itself
        tr += [p[0] for p in a]
        he += [p[0] for p in b]
    return tr, he


def scored_step(acc: Mapping[int, float], gain: Mapping[int, float], checkpoints: Sequence[int], bar: float,
                min_gain: float) -> tuple[int, bool]:  # fmt: skip
    """E124's state selection (Fable's change 1): the first checkpoint with train accuracy >= bar (E112's learn bar)
    and full-vocabulary log-odds gain >= min_gain (local_faith.MIN_GAIN), both inclusive; else the last, not learned."""
    ok = lambda s: s in acc and acc[s] >= bar and gain[s] >= min_gain
    return next(((s, True) for s in checkpoints if ok(s)), (checkpoints[-1], False))


def replicates(r_nat: Sequence[float], r_new: Sequence[float], cut: float, min_gap: float) -> dict[str, object]:
    """The filed re-gating reading: `elicit.separation` (gap over the larger seed range, >= min_gap) on the seeds' r,
    plus natural r >= cut and new-word r < cut on every seed. A nan r or a missing seed: "no call"."""
    from commute.elicit import separation

    sep = separation(list(r_nat), list(r_new), min_gap)
    if sep["verdict"] == "unreadable":
        return {"reading": "no call", **sep}
    ok = sep["verdict"] == "a_higher" and all(r >= cut for r in r_nat) and all(r < cut for r in r_new)
    return {"reading": "replicates" if ok else "not replicated", **sep}


def member_passes(verdicts: Sequence[str], predicted: str) -> bool:
    """Fable's change 6: a member-level prediction passes only if every seed's own verdict is the predicted one."""
    return len(verdicts) > 0 and all(v == predicted for v in verdicts)
